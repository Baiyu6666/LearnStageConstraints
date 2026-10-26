#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export USER_UID="$(id -u)"

docker compose down
