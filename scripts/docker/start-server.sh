#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

VATTN_SERVER_OUTPUT_DIR=${VATTN_SERVER_OUTPUT_DIR:-/tmp/vattention/${VATTN_CONTAINER_NAME}}

if ! container_exists; then
    printf 'Container does not exist yet: %s\nRun scripts/docker/create-container.sh first.\n' "${VATTN_CONTAINER_NAME}" >&2
    exit 1
fi

if ! container_running; then
    run_cmd docker start "${VATTN_CONTAINER_NAME}"
fi

readarray -t exec_args < <(docker_exec_args)

run_cmd docker exec "${exec_args[@]}" "${VATTN_CONTAINER_NAME}" bash -lc '
set -euo pipefail
output_dir="$1"
mkdir -p "$output_dir"
cd /workspace/sarathi-lean
shift
exec python -m sarathi.entrypoints.openai_server.api_server --output_dir "$output_dir" "$@"
' bash "${VATTN_SERVER_OUTPUT_DIR}" "$@"
