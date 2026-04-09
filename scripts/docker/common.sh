#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

VATTN_IMAGE_NAME=${VATTN_IMAGE_NAME:-vattention-multiuser:24.03}
VATTN_CONTAINER_NAME=${VATTN_CONTAINER_NAME:-vattn-${USER}}
VATTN_WORKSPACE_HOST=${VATTN_WORKSPACE_HOST:-${REPO_ROOT}}
VATTN_WORKSPACE_CONTAINER=${VATTN_WORKSPACE_CONTAINER:-/workspace}
VATTN_CUDA_HOST=${VATTN_CUDA_HOST:-/opt/cuda-12.1}
VATTN_LIBTORCH_HOST=${VATTN_LIBTORCH_HOST:-/opt/libtorch}
VATTN_SHM_SIZE=${VATTN_SHM_SIZE:-16g}
VATTN_TORCH_CUDA_ARCH_LIST=${VATTN_TORCH_CUDA_ARCH_LIST:-8.6}
VATTN_MAX_JOBS=${VATTN_MAX_JOBS:-4}
VATTN_HF_CACHE_HOST=${VATTN_HF_CACHE_HOST:-${HOME}/.cache/huggingface}
VATTN_PIP_CACHE_HOST=${VATTN_PIP_CACHE_HOST:-${HOME}/.cache/pip}
VATTN_TORCH_CACHE_HOST=${VATTN_TORCH_CACHE_HOST:-${HOME}/.cache/torch}

run_cmd() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'

    if [[ "${VATTN_DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi

    "$@"
}

docker_exec_args() {
    if [[ -t 0 && -t 1 ]]; then
        printf '%s\n' "-it"
    else
        printf '%s\n' "-i"
    fi
}

require_path() {
    local path="$1"
    local description="$2"

    if [[ ! -e "${path}" ]]; then
        printf 'Missing %s: %s\n' "${description}" "${path}" >&2
        exit 1
    fi
}

ensure_host_prereqs() {
    require_path "${VATTN_WORKSPACE_HOST}" "workspace mount"
    require_path "${VATTN_CUDA_HOST}" "CUDA host mount"
    require_path "${VATTN_LIBTORCH_HOST}" "libtorch host mount"

    run_cmd mkdir -p "${VATTN_HF_CACHE_HOST}" "${VATTN_PIP_CACHE_HOST}" "${VATTN_TORCH_CACHE_HOST}"
}

container_exists() {
    if [[ "${VATTN_DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    docker container inspect "${VATTN_CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
    if [[ "${VATTN_DRY_RUN:-0}" == "1" ]]; then
        return 1
    fi
    [[ "$(docker inspect -f '{{.State.Running}}' "${VATTN_CONTAINER_NAME}" 2>/dev/null || true)" == "true" ]]
}

ensure_container_running() {
    if ! container_exists; then
        printf 'Container does not exist yet: %s\nRun scripts/docker/create-container.sh first.\n' "${VATTN_CONTAINER_NAME}" >&2
        exit 1
    fi

    if ! container_running; then
        run_cmd docker start "${VATTN_CONTAINER_NAME}"
    fi
}

run_in_container() {
    local script="$1"
    readarray -t exec_args < <(docker_exec_args)
    run_cmd docker exec "${exec_args[@]}" "${VATTN_CONTAINER_NAME}" bash -lc "${script}"
}
