import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from envs import load_env
from envs.BarSponge import FEATURE_NAMES, TRAJECTORY_COLUMNS
from experiments.artifacts import dataset_fingerprint, save_run_artifacts
from experiments.bar_sponge_constraint_export import build_constraint, convert_saved_run, save_constraint
from experiments.export_saved_learned_constraints import convert


@pytest.fixture
def inputs(tmp_path):
    poses, matrices, bars, cups = [], [], [], []
    for yaw in (0., 90.):
        bar_rotation = Rotation.from_euler('z', yaw, degrees=True) * Rotation.from_euler('y', 90., degrees=True)
        bar = np.r_[[.5, -.2, 2.], bar_rotation.as_quat()]
        cup = np.array([1., 1., 0., 0., 0., 0., 1.])
        frame = Rotation.from_euler('z', yaw, degrees=True)
        relative = np.c_[.05 + .01 * np.arange(12), np.full(12, .015), np.zeros(12)]
        xyz = frame.apply(relative) + bar[:3]
        xyz[:, 2] = .31
        pose = np.c_[xyz, np.tile(frame.as_quat(), (12, 1))]
        poses.append(pose)
        matrices.append(np.c_[np.linalg.norm(xyz[:, :2] - cup[:2], axis=1), xyz[:, 2] - bar[2],
                              relative[:, 1], relative[:, 0], np.full(12, .02), np.full(12, 20.)])
        bars.append(bar)
        cups.append(cup)
    path = tmp_path / 'training.npz'
    np.savez(path, schema_version=1, trajectory=np.concatenate(poses),
             trajectory_columns=TRAJECTORY_COLUMNS, features=np.concatenate(matrices),
             feature_names=FEATURE_NAMES, timestamps=np.tile(np.arange(12) / 5., 2),
             world_z_force_n=np.full(24, 20.), tcp_speed_mps=np.full(24, .02),
             true_stage_labels=np.tile(np.repeat([0, 1, 2], 4), 2),
             true_bounds_indices=[[0, 4, 8, 12], [12, 16, 20, 24]],
             demo_bar_poses_base=bars, demo_cup_poses_base=cups, source_demo_ids=[3, 8],
             downsample_hz=5., frame_id='robot_base', bar_axis_local=[0, 0, 1],
             source_session_id='sponge_session')
    kwargs = dict(processed_demo_path=str(path), n_demos=2, source_demo_ids=[8, 3], seed=2026)
    dataset = load_env('BarSponge', **kwargs)
    semantics = np.full((3, 6), '', dtype=object)
    semantics[0, 5] = 'target_value'
    semantics[1, [1, 2, 4, 5]] = 'target_value'
    semantics[2, [0, 1]] = 'lower_bound'
    semantics[2, 5] = 'target_value'
    values = np.full((3, 6), np.nan)
    values[0, 5] = .001
    values[1, [1, 2, 4, 5]] = [.31, .015, .02, 20.5]
    values[2, [0, 1, 5]] = [.11, .33, -.017]
    constraints = dict(ConstraintFeatureNames=list(FEATURE_NAMES),
                       ConstraintLearnedSemanticsMatrix=semantics.tolist(),
                       ConstraintLearnedValueMatrix=values.tolist(),
                       ConstraintPredictedActiveMask=(semantics != '').astype(int).tolist())
    config = dict(dataset_name='BarSponge', method_name='map_balanced_pooled',
                  dataset_kwargs=kwargs, method_kwargs={'seed': 0})
    metadata = dict(dataset_name='BarSponge', method_name='map_balanced_pooled',
                    dataset_fingerprint=dataset_fingerprint(dataset), num_demos=2,
                    method_seed=0, created_at_utc='2026-10-29T08:50:00+00:00')
    return dict(dataset=dataset, constraints=constraints, predicted_cutpoints=[[4, 8], [4, 8]],
                training_config=config, training_metadata=metadata)


def test_single_file_preserves_all_modes_and_horizontal_geometry(inputs, tmp_path):
    path = save_constraint(tmp_path / 'export', **inputs)
    reject_constant = lambda value: pytest.fail('Nonstandard JSON constant: ' + value)
    payload = json.loads(path.read_text(), parse_constant=reject_constant)
    assert payload['task_id'] == 'bar_sponge'
    assert payload['stage_names'] == ['approach', 'contact_wipe', 'cup_avoid_depart']
    assert len(payload['feature_stage_modes']) == 18
    for pair in payload['feature_stage_modes']:
        column = FEATURE_NAMES.index(pair['feature_name'])
        expected = inputs['constraints']['ConstraintLearnedValueMatrix'][pair['stage']][column]
        assert pair['value'] is None if pair['mode'] == 'inactive' else pair['value'] == expected
    geometry = payload['geometry']
    np.testing.assert_allclose(geometry['stage_endpoint_poses_bar'],
                               [[.08, .015, -1.69, 0, 0, 0, 1], [.12, .015, -1.69, 0, 0, 0, 1]], atol=1e-12)
    assert [demo['source_demo_id'] for demo in geometry['per_demo']] == [8, 3]
    assert geometry['per_demo'][0]['endpoint_sample_indices'] == [3, 7]
    assert payload['provenance']['source_demo_ids'] == [8, 3]
    assert payload['provenance']['training_data_sha256'] == hashlib.sha256(
        Path(inputs['dataset'].meta['processed_demo_path']).read_bytes()).hexdigest()
    # The copied artifact remains complete after every source file is removed.
    portable = tmp_path / 'portable.json'
    portable.write_bytes(path.read_bytes())
    Path(inputs['dataset'].meta['processed_demo_path']).unlink()
    path.unlink()
    assert json.loads(portable.read_text()) == payload


@pytest.mark.parametrize('problem', ['nonfinite', 'mask', 'cuts', 'fingerprint', 'vertical_bar'])
def test_invalid_export_does_not_replace_a_valid_file(inputs, tmp_path, problem):
    output = save_constraint(tmp_path, **inputs)
    original = output.read_bytes()
    if problem == 'nonfinite':
        inputs['constraints']['ConstraintLearnedValueMatrix'][1][5] = float('nan')
    elif problem == 'mask':
        inputs['constraints']['ConstraintPredictedActiveMask'][1][5] = 0
    elif problem == 'cuts':
        inputs['predicted_cutpoints'][0] = [4, 4]
    elif problem == 'fingerprint':
        inputs['training_metadata']['dataset_fingerprint'] = 'wrong'
    else:
        inputs['dataset'].env.demo_scenes[0]['bar_pose_base'][3:] = [0, 0, 0, 1]
    with pytest.raises(ValueError):
        save_constraint(tmp_path, **inputs)
    assert output.read_bytes() == original
    assert not output.with_name('constraint.json.incomplete').exists()


@pytest.mark.parametrize('method', ['map_balanced_pooled', 'swcl'])
def test_training_automatically_exports_only_bar_sponge(inputs, tmp_path, method):
    # Other tasks must continue through their pre-existing saving branch.
    generic = tmp_path / 'other'
    result = dict(dataset=SimpleNamespace(demos=[np.zeros((4, 2))], true_cutpoints=[]),
                  pipeline='joint', joint_result={'metrics': {}})
    files = save_run_artifacts(run_dir=generic, dataset_name='OtherTask', method_name=method,
                              dataset_kwargs={'seed': 1}, method_kwargs={'seed': 0}, result=result)
    assert set(files) == {'metadata', 'config', 'metrics', 'objectives', 'segmentation', 'constraints'}
    assert not (generic / 'constraint.json').exists()
    assert json.loads((generic / 'constraints.json').read_text()) == {'true_constraints': None, 'constraint_specs': None}
    result = dict(dataset=inputs['dataset'], pipeline='joint', joint_result={
        'metrics': inputs['constraints'], 'cutpoints_hat': inputs['predicted_cutpoints'],
        'model': SimpleNamespace(num_stages=3)})
    files = save_run_artifacts(
        run_dir=tmp_path / 'BarSponge', dataset_name='BarSponge', method_name=method,
        dataset_kwargs=inputs['training_config']['dataset_kwargs'], method_kwargs={'seed': 0}, result=result)
    assert files['constraint'].name == 'constraint.json'
    assert len(json.loads(files['constraint'].read_text())['feature_stage_modes']) == 18
    assert json.loads(files['constraints'].read_text())['ConstraintFeatureNames'] == list(FEATURE_NAMES)


def test_backfill_matches_training_export_without_retraining(inputs, tmp_path):
    config = inputs['training_config']
    result = dict(dataset=inputs['dataset'], pipeline='joint', joint_result={
        'metrics': inputs['constraints'], 'cutpoints_hat': inputs['predicted_cutpoints'],
        'model': SimpleNamespace(num_stages=3)})
    files = save_run_artifacts(run_dir=tmp_path, dataset_name='BarSponge',
                              method_name='map_balanced_pooled', dataset_kwargs=config['dataset_kwargs'],
                              method_kwargs={'seed': 0}, result=result)
    original = files['constraint'].read_bytes()
    assert convert(files['constraints']).read_bytes() == original
    saved_segmentation = json.loads(files['segmentation'].read_text())
    saved_segmentation['demo_lengths'][0] += 1
    files['segmentation'].write_text(json.dumps(saved_segmentation))
    with pytest.raises(ValueError, match='demo lengths'):
        convert_saved_run(files['constraints'])
    assert files['constraint'].read_bytes() == original


def test_barclean_keeps_its_existing_learned_artifact_format(inputs, tmp_path):
    from experiments.test_artifacts import _EndpointEnv

    inputs['dataset'].env = _EndpointEnv()
    result = dict(dataset=inputs['dataset'], pipeline='joint', joint_result={
        'metrics': inputs['constraints'], 'cutpoints_hat': inputs['predicted_cutpoints'],
        'model': SimpleNamespace(num_stages=3)})
    files = save_run_artifacts(
        run_dir=tmp_path, dataset_name='BarClean', method_name='map_balanced_pooled',
        dataset_kwargs={'seed': 2026}, method_kwargs={'seed': 0}, result=result)
    assert 'constraint' not in files
    assert files['learned_constraints'].name == 'learned_constraints.json'
    assert not (tmp_path / 'constraint.json').exists()
    saved = json.loads(files['learned_constraints'].read_text())
    assert saved['schema_version'] == 5 and saved['task_id'] == 'BarClean'
    assert saved['endpoint_coordinate_frame'] == 'bar_table_task'
