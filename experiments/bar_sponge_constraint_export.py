"""Self-contained BarSponge constraints; other tasks keep their own formats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


STAGE_NAMES = ('approach', 'contact_wipe', 'cup_avoid_depart')
MODES = {'inactive', 'target_value', 'lower_bound', 'upper_bound'}
FEATURE_DEFINITIONS = {
    'cup_xy_distance': ('m', 'norm(TCP.base_xy - Cup.base_xy); center distance, no radii'),
    'bar_relative_height': ('m', 'TCP.base_z - Bar.base_z; signed height along robot-base +Z'),
    'bar_lateral_offset': ('m', '(TCP.base_xy - Bar.base_xy) dot lateral_axis'),
    'bar_longitudinal_offset': ('m', '(TCP.base_xy - Bar.base_xy) dot longitudinal_axis'),
    'tcp_speed': ('m/s', 'recorded FK translational speed trace'),
    'world_z_force': ('N', 'recorded compensated positive robot-base Z force'),
}


def _pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('BarSponge export requires finite XYZ+XYZW poses')
    norm = np.linalg.norm(pose[3:])
    if abs(norm - 1.) > .01:
        raise ValueError('BarSponge export requires normalized quaternions')
    return np.r_[pose[:3], pose[3:] / norm]


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def build_constraint(*, dataset, constraints, predicted_cutpoints,
                     training_config, training_metadata):
    """Export learned values verbatim and geometry from predicted boundaries."""
    from experiments.artifacts import dataset_fingerprint

    if (dataset.name != 'BarSponge' or training_config['dataset_name'] != 'BarSponge'
            or training_metadata['dataset_name'] != 'BarSponge'):
        raise ValueError('This single-file exporter is for BarSponge only')
    if training_config['method_name'] != training_metadata['method_name']:
        raise ValueError('Training method metadata does not match the snapshot')
    fingerprint = dataset_fingerprint(dataset)
    if fingerprint != training_metadata['dataset_fingerprint']:
        raise ValueError('Training dataset fingerprint differs from the saved run')
    names = list(constraints['ConstraintFeatureNames'])
    if not names or len(set(names)) != len(names) or set(names) - FEATURE_DEFINITIONS.keys():
        raise ValueError('BarSponge export has duplicate or unknown feature names')
    semantics = np.asarray(constraints['ConstraintLearnedSemanticsMatrix'], dtype=object)
    values = np.asarray(constraints['ConstraintLearnedValueMatrix'], dtype=float)
    active = np.asarray(constraints['ConstraintPredictedActiveMask'])
    if (semantics.shape != (3, len(names)) or values.shape != semantics.shape
            or active.shape != semantics.shape or not np.isin(active, [0, 1]).all()):
        raise ValueError('BarSponge needs matching three-stage constraint matrices')
    pairs = []
    for stage, row in enumerate(semantics):
        for column, name in enumerate(names):
            mode = str(row[column]).strip() or 'inactive'
            if mode not in MODES or bool(active[stage, column]) != (mode != 'inactive'):
                raise ValueError('Constraint mode and active mask disagree')
            value = None if mode == 'inactive' else float(values[stage, column])
            if value is not None and not np.isfinite(value):
                raise ValueError('Active BarSponge constraint value must be finite')
            pairs.append(dict(stage=stage, feature_name=name, mode=mode, value=value))
    schema = {spec['name']: spec for spec in dataset.feature_schema}
    for name in names:
        if name not in schema or schema[name]['unit'] != FEATURE_DEFINITIONS[name][0]:
            raise ValueError('BarSponge feature units differ from the export definition')

    demos = list(dataset.demos)
    ids = list(dataset.meta['source_demo_ids'])
    times = list(dataset.meta['timestamps'])
    if (not demos or predicted_cutpoints is None or len(predicted_cutpoints) != len(demos)
            or len(ids) != len(demos) or len(times) != len(demos)
            or len(set(ids)) != len(ids)):
        raise ValueError('BarSponge needs one source ID, time trace and segmentation per demo')
    if int(training_metadata['num_demos']) != len(demos):
        raise ValueError('Training demo count differs from the saved run')
    per_demo, endpoint_groups = [], [[], []]
    for ordinal, (demo, boundaries) in enumerate(zip(demos, predicted_cutpoints)):
        trajectory = np.asarray(demo, dtype=float)
        cuts = np.asarray(boundaries, dtype=float)
        stamp = np.asarray(times[ordinal], dtype=float)
        if (trajectory.ndim != 2 or trajectory.shape[1] != 7 or not np.isfinite(trajectory).all()
                or cuts.shape != (2,) or not np.isfinite(cuts).all()
                or np.any(cuts != np.floor(cuts)) or not 0 < cuts[0] < cuts[1] < len(trajectory)
                or stamp.shape != (len(trajectory),) or not np.isfinite(stamp).all()
                or np.any(np.diff(stamp) <= 0)):
            raise ValueError('Invalid predicted BarSponge boundaries or trajectory samples')
        cuts = cuts.astype(int)
        scene = dataset.env.get_demo_scene(ordinal)
        bar, cup = (_pose(scene[key]) for key in ('bar_pose_base', 'cup_pose_base'))
        axis = Rotation.from_quat(bar[3:]).apply([0., 0., 1.])
        axis[2] = 0.
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError('Bar local Z has no defined horizontal projection')
        axis /= norm
        lateral = np.cross([0., 0., 1.], axis)
        base_from_bar = Rotation.from_matrix(np.column_stack((axis, lateral, [0., 0., 1.])))
        endpoints = []
        for group, cut in zip(endpoint_groups, cuts):
            pose = _pose(trajectory[cut - 1])
            delta = pose[:3] - bar[:3]
            relative = np.r_[delta @ axis, delta @ lateral, delta[2],
                             (base_from_bar.inv() * Rotation.from_quat(pose[3:])).as_quat()]
            endpoints.append(relative.tolist())
            group.append(relative)
        per_demo.append(dict(source_demo_id=int(ids[ordinal]),
                             predicted_cutpoints=cuts.tolist(),
                             endpoint_sample_indices=(cuts - 1).tolist(),
                             endpoint_times_s=stamp[cuts - 1].tolist(),
                             bar_pose_base=bar.tolist(), cup_pose_base=cup.tolist(),
                             stage_endpoint_poses_bar=endpoints))
    aggregate = []
    for group in endpoint_groups:
        poses = np.asarray(group)
        quaternion = Rotation.from_quat(poses[:, 3:]).mean().as_quat()
        if quaternion[3] < 0:
            quaternion = -quaternion
        aggregate.append(np.r_[poses[:, :3].mean(axis=0), quaternion].tolist())
    archive_path = Path(dataset.meta['processed_demo_path'])
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    frame = str(dataset.meta['frame_id'])
    if not frame or frame != dataset.env.frame_id:
        raise ValueError('BarSponge robot-base frame is missing or inconsistent')
    session = str(dataset.meta['source_session_id'])
    if not session:
        raise ValueError('BarSponge source recording session is missing')
    method = str(training_metadata['method_name'])
    seed = int(training_metadata['method_seed'])
    payload = dict(
        schema_version=1, artifact_type='bar_sponge_stage_constraints', task_id='bar_sponge',
        id=f'bar_sponge_{method}_{seed:03d}_{session}',
        display_name=f'Bar Sponge · {method} · seed {seed} · {session}',
        created_at_utc=training_metadata['created_at_utc'],
        num_stages=3, stage_names=list(STAGE_NAMES),
        feature_schema=[dict(name=name, unit=FEATURE_DEFINITIONS[name][0],
                             definition=FEATURE_DEFINITIONS[name][1]) for name in names],
        feature_definition=dict(
            value_space='physical units; not normalized',
            evaluation_point='recorded FK end-effector origin',
            force_axis='positive robot-base Z',
            longitudinal_axis='normalize(project robot-base XY of Bar local +Z)',
            lateral_axis='robot-base +Z cross longitudinal_axis',
            scene_policy='Bar and Cup fixed in robot-base frame per task',
            cup_distance_semantics='TCP point XY to Cup center; no cup or tool radii',
            height_semantics='TCP.base_z - Bar.base_z; frozen tracked Bar origin; robot-base +Z'),
        task_frame=dict(frame_id=frame, snapshot_policy='frozen_per_task',
                        tracked_bar='baiyu_bar_force', tracked_cup='baiyu_cup'),
        feature_stage_modes=pairs,
        geometry=dict(
            endpoint_coordinate_frame='bar_center_base_axes',
            origin='tracked Bar origin in robot-base XYZ',
            axes=['longitudinal_axis', 'lateral_axis', 'robot-base +Z'],
            pose_columns=['longitudinal_m', 'lateral_m', 'bar_relative_height_m', 'qx', 'qy', 'qz', 'qw'],
            orientation_frame='bar_center_base_axes; quaternion order xyzw',
            endpoint_names=['contact_start', 'contact_end'],
            stage_endpoint_poses_bar=aggregate,
            aggregation=dict(position='arithmetic mean across selected demos',
                             orientation='Markley quaternion mean',
                             sample_policy='last sample before each predicted cutpoint'),
            per_demo=per_demo),
        provenance=dict(source_session_id=session, source_demo_ids=[int(value) for value in ids],
                        training_data_sha256=archive_sha256, dataset_fingerprint=fingerprint,
                        training_config=_plain(training_config),
                        training_metadata=_plain(training_metadata)),
    )
    # Strict JSON is checked before an existing export can be replaced.
    json.dumps(payload, allow_nan=False)
    return payload


def save_constraint(run_dir, **kwargs):
    payload = build_constraint(**kwargs)
    output = Path(run_dir) / 'constraint.json'
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = output.with_name(output.name + '.incomplete')
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(content)
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def convert_saved_run(constraints_path):
    from envs import load_env

    constraints_path = Path(constraints_path)
    run_dir = constraints_path.parent
    read = lambda name: json.loads((run_dir / name).read_text(encoding='utf-8'))
    config, metadata = read('config_snapshot.json'), read('metadata.json')
    if config['dataset_name'] != 'BarSponge' or metadata['dataset_name'] != 'BarSponge':
        raise ValueError('This saved-run converter is for BarSponge only')
    dataset = load_env('BarSponge', **config['dataset_kwargs'])
    segmentation = read('segmentation.json')
    if segmentation['demo_lengths'] != [len(demo) for demo in dataset.demos]:
        raise ValueError('Saved BarSponge demo lengths differ from the training dataset')
    return save_constraint(
        run_dir, dataset=dataset,
        constraints=json.loads(constraints_path.read_text(encoding='utf-8')),
        predicted_cutpoints=segmentation['predicted_cutpoints'],
        training_config=config, training_metadata=metadata,
    )
