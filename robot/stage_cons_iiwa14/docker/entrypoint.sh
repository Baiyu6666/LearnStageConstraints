#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
source /home/ros/ros_ws/devel/setup.bash
profile_exports="$(python3 -m stage_robot_config shell)"
eval "$profile_exports"
if [[ "$1" == "roslaunch" ]]; then
    for argument in "$@"; do
        case "$argument" in
            model:=*) expected="$STAGE_ROBOT_MODEL" ;;
            robot_name:=*) expected="$STAGE_ROBOT_NAME" ;;
            robot_ip:=*) expected="$ROBOT_IP" ;;
            *) continue ;;
        esac
        if [[ "${argument#*:=}" != "$expected" ]]; then
            echo "Launch override conflicts with the selected robot/site: $argument" >&2
            exit 1
        fi
    done
fi
if [[ "$1" == "roslaunch" && ( "$2" == "iiwa_driver" || "$2" == "stage_real_executor" ) ]]; then
    python3 -m stage_robot_config hardware
fi
if [[ "$1" == "roslaunch" ]]; then
    configured="$(rosparam get /stage_cons/runtime_identity 2>/dev/null || true)"
    identity="$STAGE_RUNTIME_IDENTITY"
    if [[ -n "$configured" && "$configured" != "$identity" ]]; then
        echo "Runtime profile changed in an active ROS graph; stop the workstation first." >&2
        exit 1
    fi
    if rosparam list >/dev/null 2>&1; then
        rosparam set /stage_cons/runtime_identity "$identity"
    fi
fi
exec "$@"
