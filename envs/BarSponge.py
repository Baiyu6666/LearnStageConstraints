"""Three-stage bar-sponge real-demonstration environment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .base import TaskBundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAMES = ('cup_xy_distance', 'bar_relative_height', 'bar_lateral_offset',
                 'bar_longitudinal_offset', 'tcp_speed', 'world_z_force')
TRAJECTORY_COLUMNS = ('ee_x', 'ee_y', 'ee_z', 'qx', 'qy', 'qz', 'qw')


class BarSpongeEnv:
    task_name = 'BarSponge'
    default_learning_features = FEATURE_NAMES

    def __init__(self, *, downsample_hz, frame_id, bar_axis_local=(0, 0, 1)):
        self.dt = 1.0 / float(downsample_hz)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('BarSponge sample rate must be positive')
        self.frame_id = str(frame_id)
        if not self.frame_id:
            raise ValueError('BarSponge requires a robot base frame')
        self.bar_axis_local = np.asarray(bar_axis_local, dtype=float)
        if self.bar_axis_local.shape != (3,) or not np.allclose(self.bar_axis_local, [0, 0, 1]):
            raise ValueError('BarSponge training expects the tracked bar local Z axis')
        self.scene = None
        self.demo_scenes = []
        self._observed_traces = {}
        self.hide_true_stage_end_markers = True
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = {}
        self.constraint_specs = []

    @staticmethod
    def get_feature_schema():
        names = FEATURE_NAMES
        units = ('m', 'm', 'm', 'm', 'm/s', 'N')
        descriptions = (
            'TCP point XY distance to the tracked cup center',
            'Signed TCP base Z minus frozen tracked Bar center base Z',
            'Signed offset along base Z cross projected bar local Z',
            'Signed TCP minus Bar origin along normalized bar local Z projected into base XY',
            'FK end-effector translational speed',
            'Compensated positive robot-base Z force',
        )
        return [{'id': i, 'column_idx': i, 'name': name, 'unit': units[i],
                 'description': descriptions[i]} for i, name in enumerate(names)]

    def get_true_constraints(self):
        return {}

    def get_constraint_specs(self):
        return []

    def get_observation_spec(self):
        return {'frame_id': self.frame_id, 'sample_period_s': self.dt,
                'trajectory_columns': TRAJECTORY_COLUMNS,
                'feature_schema': self.get_feature_schema(),
                'stage_names': ('approach', 'contact_wipe', 'cup_avoid_depart')}

    def get_demo_scene(self, demo_index):
        return self.demo_scenes[int(demo_index)]

    def set_demo_scenes(self, scenes):
        self.demo_scenes = [self._validate_scene(scene) for scene in scenes]
        self.scene = self.demo_scenes[0] if self.demo_scenes else None

    @staticmethod
    def _validate_scene(scene):
        bar = np.asarray(scene['bar_pose_base'], dtype=float)
        cup = np.asarray(scene['cup_pose_base'], dtype=float)
        if bar.shape != (7,) or cup.shape != (7,) or not np.all(np.isfinite(bar)) or not np.all(np.isfinite(cup)):
            raise ValueError('Bar and Cup poses must be finite XYZ+XYZW base-frame rows')
        if abs(np.linalg.norm(bar[3:]) - 1) > .01 or abs(np.linalg.norm(cup[3:]) - 1) > .01:
            raise ValueError('Object quaternions must be normalized')
        return {'bar_pose_base': bar.copy(), 'cup_pose_base': cup.copy()}

    @staticmethod
    def _trace_key(trajectory):
        array = np.ascontiguousarray(np.asarray(trajectory, dtype=np.float64))
        return (array.shape, hashlib.sha256(array.tobytes()).digest())

    def register_observed_traces(self, trajectory, speed, force):
        array = np.asarray(trajectory, dtype=float)
        speed = np.asarray(speed, dtype=float)
        force = np.asarray(force, dtype=float)
        if speed.shape != (len(array),) or force.shape != (len(array),):
            raise ValueError('Speed and force traces must align with the trajectory')
        self._observed_traces[self._trace_key(array)] = (speed.copy(), force.copy())

    def compute_all_features_matrix(self, trajectory, feat_ids=None, scene=None):
        pose = np.asarray(trajectory, dtype=float)
        if pose.ndim != 2 or pose.shape[1] != 7 or not np.all(np.isfinite(pose)):
            raise ValueError('BarSponge trajectory must have finite XYZ+XYZW rows')
        active_scene = self._validate_scene(scene) if scene is not None else self.scene
        if active_scene is None:
            raise ValueError('BarSponge needs frozen Bar and Cup base-frame poses')
        traces = self._observed_traces.get(self._trace_key(pose))
        if traces is None:
            raise ValueError('Measured speed and force are required for BarSponge features')
        speed, force = traces
        bar = active_scene['bar_pose_base']
        cup = active_scene['cup_pose_base']
        axis = Rotation.from_quat(bar[3:]).apply(self.bar_axis_local)[:2]
        axis /= np.linalg.norm(axis)
        lateral = np.array([-axis[1], axis[0]])
        matrix = np.column_stack((
            np.linalg.norm(pose[:, :2] - cup[:2], axis=1),
            pose[:, 2] - bar[2],
            (pose[:, :2] - bar[:2]) @ lateral,
            (pose[:, :2] - bar[:2]) @ axis,
            speed,
            force,
        ))
        return matrix if feat_ids is None else matrix[:, feat_ids]


def load_BarSponge(*, processed_demo_path, n_demos=5, seed=2026, source_demo_ids=None):
    path = Path(processed_demo_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with np.load(path, allow_pickle=False) as archive:
        required = {'schema_version', 'trajectory', 'trajectory_columns', 'features',
                    'feature_names', 'timestamps', 'world_z_force_n', 'tcp_speed_mps',
                    'true_stage_labels', 'true_bounds_indices', 'demo_bar_poses_base',
                    'demo_cup_poses_base', 'source_demo_ids', 'downsample_hz', 'frame_id',
                    'bar_axis_local'}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f'BarSponge archive is missing {missing}')
        if int(archive['schema_version']) != 1:
            raise ValueError('Unsupported BarSponge training schema')
        pose = np.asarray(archive['trajectory'], dtype=float)
        features = np.asarray(archive['features'], dtype=float)
        timestamps = np.asarray(archive['timestamps'], dtype=float)
        force = np.asarray(archive['world_z_force_n'], dtype=float)
        speed = np.asarray(archive['tcp_speed_mps'], dtype=float)
        labels = np.asarray(archive['true_stage_labels'], dtype=int)
        bounds = np.asarray(archive['true_bounds_indices'], dtype=int)
        bar_poses = np.asarray(archive['demo_bar_poses_base'], dtype=float)
        cup_poses = np.asarray(archive['demo_cup_poses_base'], dtype=float)
        source_ids = np.asarray(archive['source_demo_ids'], dtype=int)
        hz = float(archive['downsample_hz'])
        frame = str(archive['frame_id'].item())
        axis_local = np.asarray(archive['bar_axis_local'], dtype=float)
        names = tuple(str(name) for name in archive['feature_names'].tolist())
        columns = tuple(str(name) for name in archive['trajectory_columns'].tolist())
        source_session_id = str(archive['source_session_id'].item()) if 'source_session_id' in archive.files else ''
    if names != FEATURE_NAMES or columns != TRAJECTORY_COLUMNS:
        raise ValueError('BarSponge feature or trajectory columns are out of order')
    if pose.ndim != 2 or pose.shape[1] != 7 or features.shape != (len(pose), len(FEATURE_NAMES)):
        raise ValueError('BarSponge trajectory/features have invalid shapes')
    if any(values.shape != (len(pose),) for values in (timestamps, force, speed, labels)):
        raise ValueError('BarSponge sample traces are misaligned')
    if bounds.ndim != 2 or bounds.shape[1] != 4 or bar_poses.shape != (len(bounds), 7) or cup_poses.shape != (len(bounds), 7) or source_ids.shape != (len(bounds),):
        raise ValueError('BarSponge demo boundaries or scene poses have invalid shapes')
    if not all(np.all(np.isfinite(values)) for values in (pose, features, timestamps, force, speed, bar_poses, cup_poses)):
        raise ValueError('BarSponge training data contains non-finite values')
    if not np.allclose(features[:, 4], speed) or not np.allclose(features[:, 5], force):
        raise ValueError('Stored speed or force feature does not match its source trace')
    if bounds[0, 0] != 0 or bounds[-1, -1] != len(pose) or np.any(bounds[1:, 0] != bounds[:-1, -1]) or np.any(np.diff(bounds, axis=1) < 3):
        raise ValueError('BarSponge stage bounds must be contiguous with three nonempty stages')

    if source_demo_ids is None:
        if int(n_demos) < 1 or int(n_demos) > len(bounds):
            raise ValueError(f'Expected 1..{len(bounds)} demos, got {n_demos}')
        selected_indices = list(range(int(n_demos)))
    else:
        requested_ids = [int(value) for value in source_demo_ids]
        if not requested_ids:
            raise ValueError('source_demo_ids must contain at least one demo ID')
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError(f'source_demo_ids contains duplicates: {requested_ids}')
        if len(set(source_ids.tolist())) != len(source_ids):
            raise ValueError('Processed BarSponge data contains duplicate source_demo_ids')
        id_to_index = {int(source_id): index for index, source_id in enumerate(source_ids)}
        missing_ids = [source_id for source_id in requested_ids if source_id not in id_to_index]
        if missing_ids:
            raise ValueError(
                f'Requested source_demo_ids {missing_ids} are not present in {path}; '
                f'available IDs are {source_ids.tolist()}'
            )
        selected_indices = [id_to_index[source_id] for source_id in requested_ids]

    env = BarSpongeEnv(downsample_hz=hz, frame_id=frame, bar_axis_local=axis_local)
    scenes = [{'bar_pose_base': bar_poses[i], 'cup_pose_base': cup_poses[i]}
              for i in selected_indices]
    env.set_demo_scenes(scenes)
    demos, matrices, true_labels, true_cutpoints, demo_times = [], [], [], [], []
    for index, archive_index in enumerate(selected_indices):
        start, first, second, end = bounds[archive_index]
        trajectory = pose[start:end].copy()
        matrix = features[start:end].copy()
        observed = labels[start:end]
        expected = np.repeat(np.arange(3), [first-start, second-first, end-second])
        if not np.array_equal(observed, expected):
            raise ValueError(f'BarSponge demo {index} true-stage labels disagree with bounds')
        env.register_observed_traces(trajectory, speed[start:end], force[start:end])
        if not np.allclose(env.compute_all_features_matrix(trajectory, scene=scenes[index]), matrix, atol=1e-6):
            raise ValueError(f'BarSponge demo {index} feature values disagree with base-frame scene geometry')
        demos.append(trajectory)
        matrices.append(matrix)
        true_labels.append(expected)
        true_cutpoints.append(np.asarray([first-start, second-start], dtype=int))
        demo_times.append(timestamps[start:end].copy())
    return TaskBundle(
        name=env.task_name, demos=demos, features=matrices, env=env,
        true_cutpoints=true_cutpoints, true_labels=true_labels,
        feature_schema=env.get_feature_schema(), true_constraints=None,
        constraint_specs=None,
        meta={'seed': int(seed), 'data_source': 'processed_real_demo',
              'processed_demo_path': str(path), 'source_session_id': source_session_id,
              'source_demo_ids': source_ids[selected_indices].tolist(),
              'stage_names': ['approach', 'contact_wipe', 'cup_avoid_depart'],
              'frame_id': frame, 'timestamps': demo_times,
              'scene_specs': [{'bar_pose_base': scene['bar_pose_base'].tolist(),
                               'cup_pose_base': scene['cup_pose_base'].tolist()}
                              for scene in scenes],
              'observation_specs': env.get_observation_spec(),
              'default_learning_features': list(env.default_learning_features)},
    )
