#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ensure_container_running

MODE=${1:-paged}
shift || true

readarray -t exec_args < <(docker_exec_args)
run_cmd docker exec "${exec_args[@]}" "${VATTN_CONTAINER_NAME}" \
    bash -lc "cd ${VATTN_WORKSPACE_CONTAINER} && python scripts/deepseek_scaffold_smoke.py --mode ${MODE} $*"
