import numpy as np
import pytest

from envs import load_env
from envs.BarSponge import FEATURE_NAMES, TRAJECTORY_COLUMNS


@pytest.fixture
def archive_path(tmp_path):
    count = 27
    trajectory = np.zeros((count, 7))
    trajectory[:, 0] = np.arange(count) / 100
    trajectory[:, 1] = np.repeat([0.1, 0.2, 0.3], 9)
    trajectory[:, 2] = 0.5
    trajectory[:, 6] = 1
    bars = np.zeros((3, 7))
    bars[:, 0] = [0.1, 0.2, 0.3]
    bars[:, 4] = bars[:, 6] = np.sqrt(0.5)
    cups = np.zeros((3, 7))
    cups[:, 0] = [1, 2, 3]
    cups[:, 6] = 1
    speed = np.arange(count) / 1000
    force = np.arange(count, dtype=float)
    features = np.column_stack((
        np.linalg.norm(trajectory[:, :2] - np.repeat(cups[:, :2], 9, axis=0), axis=1),
        trajectory[:, 2], trajectory[:, 1], speed, force,
    ))
    path = tmp_path / 'training.npz'
    np.savez(
        path, schema_version=1, trajectory=trajectory,
        trajectory_columns=TRAJECTORY_COLUMNS, features=features,
        feature_names=FEATURE_NAMES, timestamps=np.arange(count) / 5,
        world_z_force_n=force, tcp_speed_mps=speed,
        true_stage_labels=np.tile(np.repeat([0, 1, 2], 3), 3),
        true_bounds_indices=[[0, 3, 6, 9], [9, 12, 15, 18], [18, 21, 24, 27]],
        demo_bar_poses_base=bars, demo_cup_poses_base=cups,
        source_demo_ids=[11, 22, 33], downsample_hz=5,
        frame_id='robot_base', bar_axis_local=[0, 0, 1],
    )
    return path


def test_selection_preserves_requested_order_and_aligned_data(archive_path):
    full = load_env('BarSponge', processed_demo_path=archive_path, n_demos=3)
    selected = load_env(
        'BarSponge', processed_demo_path=archive_path,
        source_demo_ids=[33, 11], n_demos=5,
    )
    assert selected.meta['source_demo_ids'] == [33, 11]
    assert len(selected.demos) == 2
    for index, original in enumerate([2, 0]):
        for field in ('demos', 'features', 'true_cutpoints', 'true_labels'):
            np.testing.assert_array_equal(getattr(selected, field)[index], getattr(full, field)[original])
        np.testing.assert_array_equal(selected.meta['timestamps'][index], full.meta['timestamps'][original])
        assert selected.meta['scene_specs'][index] == full.meta['scene_specs'][original]
        np.testing.assert_allclose(
            selected.env.compute_all_features_matrix(
                selected.demos[index], scene=selected.env.get_demo_scene(index),
            ), selected.features[index],
        )


@pytest.mark.parametrize('ids, message', [
    ([], 'at least one'), ([11, 11], 'duplicates'), ([99], 'available IDs'),
])
def test_invalid_demo_selection(archive_path, ids, message):
    with pytest.raises(ValueError, match=message):
        load_env('BarSponge', processed_demo_path=archive_path, source_demo_ids=ids)


def test_default_selection_uses_first_n_demos(archive_path):
    bundle = load_env('BarSponge', processed_demo_path=archive_path, n_demos=2)
    assert bundle.meta['source_demo_ids'] == [11, 22]
