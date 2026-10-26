"""Access the shared runtime profiles from host-side experiment code."""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "robot/stage_cons_iiwa14/ros_ws/src/stage_robot_config/src"))
from stage_robot_config import RuntimeConfig, active, resolve
