import json

import numpy as np
import pytest

from envs import load_env
from envs.BarSponge import BarSpongeEnv, FEATURE_NAMES, TRAJECTORY_COLUMNS
from experiments.bar_sponge_multiset import combinations, merge_sets


def source_archive(path, scene_shift, frame='base'):
    trajectory = np.zeros((9, 7))
    trajectory[:, 0] = scene_shift + np.arange(9) / 100
    trajectory[:, 1] = .1
    trajectory[:, 2] = .3
    trajectory[:, 6] = 1
    bar = np.array([scene_shift, 0., 0., 0., np.sqrt(.5), 0., np.sqrt(.5)])
    cup = np.array([scene_shift + .5, -.2, 0., 0., 0., 0., 1.])
    scene = dict(bar_pose_base=bar, cup_pose_base=cup)
    env = BarSpongeEnv(downsample_hz=5., frame_id=frame)
    speed = np.full(9, .02)
    force = np.repeat([0., 20., 0.], 3)
    env.register_observed_traces(trajectory, speed, force)
    features = env.compute_all_features_matrix(trajectory, scene=scene)
    np.savez(path, schema_version=1, trajectory=trajectory, trajectory_columns=TRAJECTORY_COLUMNS,
             features=features, feature_names=FEATURE_NAMES, timestamps=np.arange(9) / 5,
             joint_positions=np.zeros((9, 7)), world_z_force_n=force, tcp_speed_mps=speed,
             source_demo_ids=[1], true_stage_labels=np.repeat([0, 1, 2], 3),
             true_bounds_indices=[[0, 3, 6, 9]], demo_bar_poses_base=[bar], demo_cup_poses_base=[cup],
             downsample_hz=5., frame_id=frame, bar_axis_local=[0, 0, 1],
             stage_names=['approach', 'contact', 'depart'], boundary_rule='force',
             source_session_id=f'session_{scene_shift}')


def test_merge_preserves_distinct_scenes_samples_and_source_identity(tmp_path):
    sources = [tmp_path / 'first.npz', tmp_path / 'second.npz']
    for source, shift in zip(sources, [0., .2]):
        source_archive(source, shift)
    output = tmp_path / 'merged.npz'
    manifest = merge_sets(sources, output)
    bundle = load_env('BarSponge', processed_demo_path=output, source_demo_ids=[101, 1])
    assert bundle.meta['source_demo_ids'] == [101, 1]
    for index, source in enumerate(reversed(sources)):
        original = load_env('BarSponge', processed_demo_path=source, n_demos=1)
        for field in ('demos', 'features', 'true_labels', 'true_cutpoints'):
            np.testing.assert_array_equal(getattr(bundle, field)[index], getattr(original, field)[0])
        assert bundle.meta['scene_specs'][index] == original.meta['scene_specs'][0]
        np.testing.assert_array_equal(bundle.meta['timestamps'][index], original.meta['timestamps'][0])
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['true_bounds_indices'], [[0, 3, 6, 9], [9, 12, 15, 18]])
        np.testing.assert_array_equal(archive['demo_id'], np.repeat([1, 101], 9))
        np.testing.assert_array_equal(archive['source_original_demo_ids'], [1, 1])
        np.testing.assert_array_equal(archive['source_set_ids'], [0, 1])
    assert manifest['demo_count'] == 2 and manifest['sample_count'] == 18
    assert json.loads(output.with_suffix('.json').read_text()) == manifest


def test_incompatible_frames_are_rejected_before_output(tmp_path):
    sources = [tmp_path / 'first.npz', tmp_path / 'second.npz']
    source_archive(sources[0], 0.)
    source_archive(sources[1], .2, frame='different_base')
    with pytest.raises(ValueError, match='Incompatible frame_id'):
        merge_sets(sources, tmp_path / 'merged.npz')
    assert not (tmp_path / 'merged.npz').exists()


def test_combinations_cover_every_setting_and_respect_demo_limits():
    groups = [[100 * group + i for i in range(1, 5)] for group in range(5)]
    sampled = list(combinations(groups, 60, 2026))
    assert len(set(sampled)) == 60
    assert sampled == list(combinations(groups, 60, 2026))
    for row in sampled:
        assert 8 <= len(row) <= 12
        assert all(1 <= sum(i // 100 == group for i in row) <= 3 for group in range(5))
