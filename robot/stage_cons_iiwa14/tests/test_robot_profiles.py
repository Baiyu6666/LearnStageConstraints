import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as bullet

WORKSTATION = Path(__file__).resolve().parents[1]
PROJECT = WORKSTATION.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(WORKSTATION / "ros_ws/src/stage_robot_config/src"))
import stage_robot_config as profiles


class RobotProfileTest(unittest.TestCase):
    def config(self, model):
        return profiles.RuntimeConfig(
            robot="iiwa" + str(model),
            site="lab_iiwa14" if model == 14 else "iiwa7_new",
            environ={},
        )

    def test_profile_site_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "belongs to"):
            profiles.RuntimeConfig(robot="iiwa7", site="lab_iiwa14", environ={})

    def test_unknown_or_traversal_profile_rejected(self):
        for identifier in ("../iiwa7", "iiwa99"):
            with self.assertRaises((ValueError, FileNotFoundError)):
                profiles.RuntimeConfig(robot=identifier, site="iiwa7_new", environ={})

    def test_uncommissioned_site_blocks_hardware(self):
        cfg = self.config(7)
        cfg.site["hardware_commissioned"] = False
        with self.assertRaisesRegex(RuntimeError, "not commissioned"):
            cfg.require_hardware()
        self.config(14).require_hardware()

    def test_commission_flag_does_not_bypass_missing_network(self):
        cfg = self.config(7)
        cfg.site["hardware_commissioned"] = True
        with self.assertRaisesRegex(RuntimeError, "Set host_ip"):
            cfg.require_hardware()

    def test_robot_and_host_network_validation(self):
        cfg = self.config(14)
        for robot_ip in ("192.170.10.1", "192.170.11.4", "192.170.10.255"):
            cfg.site["network"]["robot_ip"] = robot_ip
            with self.assertRaises(ValueError):
                cfg.check_network()

    def test_every_safety_interface_resolves_to_selected_namespace(self):
        for model in (7, 14):
            cfg = self.config(model)
            for suffix in ("iiwa_driver/set_position_commanding",
                           "iiwa_driver/position_command_heartbeat", "real_executor/abort",
                           "real_executor/prepare", "real_executor/execute",
                           "PositionTrajectoryController/follow_joint_trajectory"):
                self.assertEqual(cfg.resolve("/iiwa14/" + suffix), cfg.topic(suffix))
            self.assertEqual(cfg.resolve("iiwa14_joint_7"), cfg.robot["joint_names"][-1])
            self.assertEqual(cfg.resolve("iiwa14_link_0"), cfg.robot["base_frame"])

    def test_tracking_base_is_independent_of_robot_namespace(self):
        cfg = self.config(7)
        cfg.tracking.update(base="new_base", object="new_bar")
        self.assertEqual(cfg.tracking_topic("object"), "/vrpn_client_node/new_bar/pose_from_new_base")
        self.assertEqual(cfg.name, "iiwa7")
        self.assertEqual(cfg.resolve("/vrpn_client_node/baiyu_bar/pose_from_iiwa14"),
                         cfg.tracking_topic("object"))

    def test_site_paths_and_kinematics_are_distinct(self):
        a, b = self.config(14), self.config(7)
        self.assertNotEqual(a.task_dir, b.task_dir)
        self.assertNotEqual(a.scene_path, b.scene_path)
        self.assertNotEqual(a.robot["joint_origins_xyz"], b.robot["joint_origins_xyz"])

    def test_forward_kinematics_matches_pybullet_from_each_actual_xacro(self):
        """Build an independent joint chain from each model's xacro origins."""
        from experiments import extract_bar_inspection_rosbag as extraction
        for model in (7, 14):
            cfg = self.config(model)
            xml = ET.parse(WORKSTATION / ("ros_ws/src/iiwa_description/urdf/iiwa%d.xacro" % model))
            joints = {e.get("name"): e for e in xml.iter("joint")}
            def vector(text):
                values = re.findall(r"\$\{[^}]+\}|[^\s]+", text)
                return [float(eval(v[2:-1], {"__builtins__": {}}, {"PI": math.pi}))
                        if v.startswith("${") else float(v) for v in values]
            urdf = ['<robot name="test"><link name="link0"/>']
            for i in range(1, 8):
                original = joints["${robot_name}_joint_%d" % i]
                xyz = vector(original.find("origin").get("xyz"))
                rpy = vector(original.find("origin").get("rpy"))
                np.testing.assert_allclose(xyz, cfg.robot["joint_origins_xyz"][i-1], atol=1e-12)
                np.testing.assert_allclose(rpy, cfg.robot["joint_origins_rpy"][i-1], atol=1e-12)
                urdf.append('<link name="link%d"/><joint name="joint%d" type="revolute">'
                            '<parent link="link%d"/><child link="link%d"/>'
                            '<origin xyz="%s" rpy="%s"/><axis xyz="0 0 1"/>'
                            '<limit lower="-3.1" upper="3.1" effort="100" velocity="1"/>'
                            '</joint>' % (i, i, i-1, i, ' '.join(map(str, xyz)), ' '.join(map(str, rpy))))
            urdf.append('</robot>')
            with tempfile.TemporaryDirectory() as directory:
                filename = Path(directory) / "chain.urdf"
                filename.write_text(''.join(urdf))
                client = bullet.connect(bullet.DIRECT)
                try:
                    robot = bullet.loadURDF(str(filename), useFixedBase=True, physicsClientId=client)
                    samples = np.random.default_rng(2026).uniform(-1, 1, (5, 7))
                    with mock.patch.object(extraction, "JOINT_ORIGINS_XYZ", np.array(cfg.robot["joint_origins_xyz"])), \
                         mock.patch.object(extraction, "JOINT_ORIGINS_RPY", np.array(cfg.robot["joint_origins_rpy"])):
                        poses = extraction.robot_fk(samples)
                    for sample, pose in zip(samples, poses):
                        for index, angle in enumerate(sample):
                            bullet.resetJointState(robot, index, angle, physicsClientId=client)
                        state = bullet.getLinkState(robot, 6, computeForwardKinematics=True, physicsClientId=client)
                        np.testing.assert_allclose(pose[:3], state[4], atol=1e-6)
                        self.assertAlmostEqual(abs(np.dot(pose[3:], state[5])), 1.0, places=6)
                finally:
                    bullet.disconnect(client)

    def test_gui_uses_selected_site_and_hardware_start_is_blocked(self):
        cfg = self.config(7)
        cfg.site["hardware_commissioned"] = False
        with mock.patch.object(profiles, "_active", cfg):
            spec = importlib.util.spec_from_file_location("iiwa7_host_test", WORKSTATION / "host_gui/supervisor.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.SCENE_CONFIG, cfg.scene_path)
            self.assertEqual(module.TASK_CONFIGS["BarClean"], cfg.task_dir / "bar_clean_true.json")
            subject = object.__new__(module.Supervisor)
            with self.assertRaisesRegex(RuntimeError, "not commissioned"):
                subject._start_real_station()

    def test_startup_environment_is_not_changed_by_later_env_edits(self):
        cfg = self.config(14)
        with mock.patch.dict(os.environ, {"ROBOT_PROFILE": "iiwa7", "SITE_PROFILE": "iiwa7_new"}):
            self.assertEqual(cfg.environment()["ROBOT_PROFILE"], "iiwa14")
            self.assertEqual(cfg.environment()["SITE_PROFILE"], "lab_iiwa14")

    def test_host_rejects_a_container_with_a_different_profile(self):
        cfg = self.config(14)
        with mock.patch.object(profiles, "_active", cfg):
            spec = importlib.util.spec_from_file_location("profile_mismatch_test", WORKSTATION / "host_gui/supervisor.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            subject = object.__new__(module.Supervisor)
            subject._run = mock.Mock(return_value=mock.Mock(stdout=json.dumps(
                ["ROBOT_PROFILE=iiwa7", "SITE_PROFILE=iiwa7_new"])))
            with self.assertRaisesRegex(RuntimeError, "profiles differ"):
                subject._assert_selected_container_profile()

    def test_learning_configuration_and_model_outputs_are_site_specific(self):
        from runners import run_benchmark as runner
        cfg = self.config(7)
        with mock.patch.object(runner, "robot_runtime", return_value=cfg):
            data = runner._load_env_config(PROJECT / "configs", "BarClean")
            self.assertIn("iiwa7/iiwa7_new", data["processed_demo_path"])
            with mock.patch.object(runner, "run_benchmark", return_value={"results": [], "dataset_seeds": []}) as run, \
                 mock.patch.object(sys, "argv", ["run_benchmark", "--datasets", "BarClean", "--methods", "map_balanced_pooled"]):
                runner.main()
            self.assertEqual(run.call_args.kwargs["run_output_root"], "outputs/iiwa7/iiwa7_new")
            self.assertEqual(run.call_args.kwargs["method_seeds"], [0])

    def test_entrypoint_rejects_conflicting_model_namespace_and_ip(self):
        entrypoint = (WORKSTATION / "docker/entrypoint.sh").read_text()
        start = 'if [[ "$1" == "roslaunch" ]]; then'
        guard = start + entrypoint.split(start, 1)[1].split('\nfi', 1)[0] + '\nfi\n'
        env = dict(os.environ, STAGE_ROBOT_MODEL="7", STAGE_ROBOT_NAME="iiwa7", ROBOT_IP="192.170.10.4")
        for argument in ("model:=14", "robot_name:=iiwa14", "robot_ip:=192.170.10.5"):
            result = subprocess.run(["bash", "-c", guard, "test", "roslaunch", "iiwa_driver", argument],
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("conflicts", result.stderr)
        result = subprocess.run(["bash", "-c", guard, "test", "roslaunch", "iiwa_driver",
                                 "model:=7", "robot_name:=iiwa7", "robot_ip:=192.170.10.4"],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
