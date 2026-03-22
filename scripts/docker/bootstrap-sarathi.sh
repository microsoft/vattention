#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ensure_container_running

run_in_container "
set -euo pipefail
cd /workspace
python -m pip install --no-build-isolation --no-deps -e /workspace/sarathi-lean
"
