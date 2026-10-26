#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env. Copy .env.example to .env and select robot/site profiles and review site.json." >&2
    exit 1
fi
source .env

SCENE_POSE_SOURCE="${SCENE_POSE_SOURCE:-fixed}"
[[ "${SCENE_POSE_SOURCE}" == "fixed" || "${SCENE_POSE_SOURCE}" == "optitrack" ]] || {
    echo "SCENE_POSE_SOURCE must be fixed or optitrack." >&2
    exit 1
}
USE_FIXED_SCENE=false
[[ "${SCENE_POSE_SOURCE}" == "fixed" ]] && USE_FIXED_SCENE=true

if ! docker ps --format '{{.Names}}' | grep -qx 'stage_cons_iiwa14'; then
    echo "stage_cons_iiwa14 is not running. Run ./scripts/start.sh first." >&2
    exit 1
fi

exec docker exec -it stage_cons_iiwa14 /entrypoint.sh \
    roslaunch stage_demo_recorder demo_collection.launch \
    "use_fixed_scene:=${USE_FIXED_SCENE}"
