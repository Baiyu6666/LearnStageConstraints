#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and verify the interface and IPs." >&2
    exit 1
fi
source .env

profile_python="${STAGE_HOST_PYTHON:-/usr/bin/python3}"
profile_path="$PWD/ros_ws/src/stage_robot_config/src${PYTHONPATH:+:$PYTHONPATH}"
profile_exports="$(PYTHONPATH="$profile_path" "$profile_python" -m stage_robot_config shell)"
eval "$profile_exports"
PYTHONPATH="$profile_path" "$profile_python" -m stage_robot_config network

: "${ROBOT_IFACE:?ROBOT_IFACE is required}"
: "${FRI_HOST_IP:?FRI_HOST_IP is required}"
: "${ROBOT_IP:?ROBOT_IP is required}"

container=stage_cons_iiwa14

if docker ps --format '{{.Names}}' | grep -qx 'kuka14'; then
    echo "Refusing to connect while the old dual-arm kuka14 container is running." >&2
    exit 1
fi
if ! docker ps --format '{{.Names}}' | grep -qx "${container}"; then
    echo "${container} is not running. Run ./scripts/start.sh first." >&2
    exit 1
fi
if ! ip link show "${ROBOT_IFACE}" >/dev/null 2>&1; then
    echo "Robot interface not found: ${ROBOT_IFACE}" >&2
    exit 1
fi

if ! ip -4 addr show dev "${ROBOT_IFACE}" | grep -Fq "${FRI_HOST_IP}/${FRI_HOST_PREFIX}"; then
    echo "Adding ${FRI_HOST_IP}/${FRI_HOST_PREFIX} to host interface ${ROBOT_IFACE}." >&2
    if [[ ${EUID} -eq 0 ]]; then
        ip addr add "${FRI_HOST_IP}/${FRI_HOST_PREFIX}" dev "${ROBOT_IFACE}"
    else
        sudo ip addr add "${FRI_HOST_IP}/${FRI_HOST_PREFIX}" dev "${ROBOT_IFACE}"
    fi
fi
if [[ ${EUID} -eq 0 ]]; then
    ip link set "${ROBOT_IFACE}" up
else
    sudo ip link set "${ROBOT_IFACE}" up
fi
ping -I "${ROBOT_IFACE}" -c 2 -W 1 "${ROBOT_IP}"

echo "Robot NIC configured on the host. The host-network container can now reach it."
echo "This does not start iiwa_driver or any torque controller."
