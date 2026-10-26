"""Startup-only robot/site selection. No ROS or third-party dependencies."""

import json
import hashlib
import ipaddress
import math
import os
from pathlib import Path
import re
import shlex


def workstation_root():
    return Path(__file__).resolve().parents[5]


def read_env(path):
    values = {}
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


class RuntimeConfig:
    def __init__(self, root=None, robot=None, site=None, environ=None):
        env = dict(os.environ if environ is None else environ)
        local = read_env(workstation_root() / ".env")
        local.update(env)
        self.root = Path(root or local.get("RUNTIME_CONFIG_ROOT") or
                         workstation_root() / "config/runtime").resolve()
        self.robot_id = robot or local.get("ROBOT_PROFILE", "iiwa14")
        self.site_id = site or local.get("SITE_PROFILE", "lab_iiwa14")
        for name in (self.robot_id, self.site_id):
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                raise ValueError("Invalid profile identifier")
        self.robot = json.loads((self.root / "robots" / (self.robot_id + ".json")).read_text())
        self.site_dir = self.root / "sites" / self.site_id
        self.site = json.loads((self.site_dir / "site.json").read_text())
        self.fingerprint = hashlib.sha256(json.dumps(
            {"robot": self.robot, "site": self.site}, sort_keys=True
        ).encode()).hexdigest()
        if self.site["robot_profile"] != self.robot_id:
            raise ValueError("Site {} belongs to {}, not {}".format(
                self.site_id, self.site["robot_profile"], self.robot_id))
        if self.robot["model"] not in (7, 14) or self.robot_id != "iiwa" + str(self.robot["model"]):
            raise ValueError("Unsupported robot model")
        self.name = self.robot["namespace"]
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", self.name):
            raise ValueError("Invalid robot namespace")
        if self.robot["joint_names"] != [self.name + "_joint_" + str(i) for i in range(1, 8)]:
            raise ValueError("Joint names must match the robot namespace")
        for key in ("joint_origins_xyz", "joint_origins_rpy"):
            values = self.robot[key]
            if len(values) != 7 or any(len(row) != 3 or
                not all(math.isfinite(float(v)) for v in row) for row in values):
                raise ValueError("Invalid " + key)
        for key in ("velocity_scale", "acceleration_limit_rad_s2", "max_demo_torque_nm",
                    "max_sim_joint_force_nm"):
            if not math.isfinite(float(self.robot[key])) or self.robot[key] <= 0:
                raise ValueError("Invalid " + key)
        thresholds = self.robot["external_torque_thresholds_nm"]
        if len(thresholds) != 7 or any(not math.isfinite(float(v)) or v <= 0 for v in thresholds):
            raise ValueError("Invalid external torque thresholds")
        self.tracking = dict(self.site["optitrack"])
        self.scene = json.loads(self.scene_path.read_text())
        if self.scene["frame_id"] != self.robot["base_frame"]:
            raise ValueError("Scene frame does not match the selected robot")

    @property
    def scene_path(self):
        return self.site_dir / "demo_scene.json"

    @property
    def task_dir(self):
        return self.site_dir / "tasks"

    def topic(self, suffix):
        return "/" + self.name + "/" + suffix.lstrip("/")

    def tracking_topic(self, role):
        return "/vrpn_client_node/{}/pose_from_{}".format(
            self.tracking[role], self.tracking["base"])

    def resolve(self, literal):
        """Resolve historical defaults to the selected startup contract."""
        mapping = {
            "/vrpn_client_node/baiyu_bar/pose_from_iiwa14": self.tracking_topic("object"),
            "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14": self.tracking_topic("obstacle_a"),
            "/vrpn_client_node/baiyu_obs_bar_b/pose_from_iiwa14": self.tracking_topic("obstacle_b"),
        }
        for before, after in mapping.items():
            literal = literal.replace(before, after)
        return literal.replace("/iiwa14/", "/" + self.name + "/").replace(
            "iiwa14_link_", self.name + "_link_").replace(
            "iiwa14_joint_", self.name + "_joint_")

    def require_hardware(self):
        if self.site.get("hardware_commissioned") is not True:
            raise RuntimeError("Site {} is not commissioned for {} hardware; configure and "
                               "validate the site before enabling hardware".format(self.site_id, self.robot_id))
        try:
            self.check_network()
        except ValueError as error:
            raise RuntimeError("Invalid FRI network for site {}: {}".format(self.site_id, error)) from error

    def check_network(self):
        network = self.site["network"]
        if not network.get("host_ip") or not network.get("robot_ip"):
            raise ValueError("Set host_ip and robot_ip in the selected site.json")
        host = ipaddress.IPv4Address(network["host_ip"])
        robot = ipaddress.IPv4Address(network["robot_ip"])
        prefix = network["host_prefix"]
        if not isinstance(prefix, int) or not 1 <= prefix <= 30:
            raise ValueError("FRI host_prefix must be between 1 and 30")
        subnet = ipaddress.IPv4Network("{}/{}".format(host, prefix), strict=False)
        forbidden = (subnet.network_address, subnet.broadcast_address)
        if host == robot or host not in subnet or robot not in subnet or host in forbidden or robot in forbidden:
            raise ValueError("FRI host/robot must be distinct usable addresses in the same subnet")
        if host.is_loopback or robot.is_loopback or host.is_multicast or robot.is_multicast:
            raise ValueError("FRI requires unicast robot-network addresses")

    def metadata(self):
        return {"robot_profile": self.robot_id, "site_profile": self.site_id,
                "robot_model": self.robot["model"], "robot_namespace": self.name,
                "runtime_fingerprint": self.fingerprint}

    def environment(self):
        rotation = self.scene["optitrack_to_robot"]["rotation"]
        translation = self.scene["optitrack_to_robot"].get("translation", [0, 0, 0])
        values = {"ROBOT_PROFILE": self.robot_id, "SITE_PROFILE": self.site_id,
                  "ROBOT_IP": self.site["network"].get("robot_ip") or "",
                  "FRI_HOST_IP": self.site["network"].get("host_ip") or "",
                  "FRI_HOST_PREFIX": self.site["network"]["host_prefix"],
                  "STAGE_ROBOT_MODEL": self.robot["model"], "STAGE_ROBOT_NAME": self.name,
                  "STAGE_BASE_FRAME": self.robot["base_frame"], "STAGE_EE_FRAME": self.robot["ee_frame"],
                  "STAGE_TIP_FRAME": self.robot["tip_link"], "STAGE_SCENE_ROTATION": json.dumps(rotation),
                  "STAGE_SCENE_TRANSLATION": json.dumps(translation),
                  "STAGE_RUNTIME_IDENTITY": self.robot_id + ":" + self.site_id + ":" + self.fingerprint,
                  "STAGE_BAR_TOPIC": self.tracking_topic("object"),
                  "STAGE_BAR_OUTLINE_U": json.dumps(self.scene["bar"]["outline_u"]),
                  "STAGE_BAR_OUTLINE_V": json.dumps(self.scene["bar"]["outline_v"]),
                  "STAGE_ASSISTANCE_CONFIG": str(self.site_dir / "assistance.yaml"),
                  "STAGE_SCENE_CONFIG": str(self.scene_path), "STAGE_TASK_DIR": str(self.task_dir),
                  "STAGE_SCENE_POSE_SOURCE": os.environ.get("SCENE_POSE_SOURCE", "fixed"),
                  "STAGE_FRI_CPU": self.site["fri_cpu_affinity"],
                  "STAGE_VELOCITY_SCALE": self.robot["velocity_scale"],
                  "STAGE_ACCELERATION_LIMIT": self.robot["acceleration_limit_rad_s2"],
                  "STAGE_DEMO_TORQUE_LIMIT": self.robot["max_demo_torque_nm"]}
        for key, value in self.tracking.items():
            values["STAGE_TRACKING_" + key.upper()] = value
        return {key: str(value) for key, value in values.items()}


_active = None


def active():
    global _active
    if _active is None:
        _active = RuntimeConfig()
    return _active


def resolve(literal):
    return active().resolve(literal)


def shell_exports():
    return "\n".join("export {}={}".format(k, shlex.quote(v))
                     for k, v in active().environment().items())
