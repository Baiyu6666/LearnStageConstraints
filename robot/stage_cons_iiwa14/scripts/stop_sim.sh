#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export USER_UID="$(id -u)"

docker compose \
    -p stage_cons_iiwa14_sim \
    -f compose.yaml \
    -f compose.sim.yaml \
    down
