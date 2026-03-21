#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

if ! container_exists; then
    printf 'Container does not exist yet: %s\nRun scripts/docker/create-container.sh first.\n' "${VATTN_CONTAINER_NAME}" >&2
    exit 1
fi

if ! container_running; then
    run_cmd docker start "${VATTN_CONTAINER_NAME}"
fi

readarray -t exec_args < <(docker_exec_args)
run_cmd docker exec "${exec_args[@]}" "${VATTN_CONTAINER_NAME}" bash
