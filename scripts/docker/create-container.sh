#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ensure_host_prereqs

if [[ "${VATTN_DRY_RUN:-0}" != "1" ]] && container_exists; then
    printf 'Container already exists: %s\n' "${VATTN_CONTAINER_NAME}"
    exit 0
fi

run_cmd docker run -d \
    --name "${VATTN_CONTAINER_NAME}" \
    --gpus all \
    --network host \
    --shm-size "${VATTN_SHM_SIZE}" \
    -e LIBTORCH_PATH=/opt/libtorch \
    -e PYTORCH_SKIP_VERSION_CHECK=1 \
    -e PYTHONPATH=/workspace/sarathi-lean:/workspace/vattention:/workspace/pod_attn:/workspace/sarathi-lean/sarathi \
    -e CXXFLAGS=-D_GLIBCXX_USE_CXX11_ABI=1 \
    -e TORCH_CUDA_ARCH_LIST="${VATTN_TORCH_CUDA_ARCH_LIST}" \
    -e MAX_JOBS="${VATTN_MAX_JOBS}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e TORCH_HOME=/root/.cache/torch \
    -e PIP_CACHE_DIR=/root/.cache/pip \
    -v "${VATTN_WORKSPACE_HOST}:${VATTN_WORKSPACE_CONTAINER}" \
    -v "${VATTN_LIBTORCH_HOST}:/opt/libtorch:ro" \
    -v "${VATTN_CUDA_HOST}:/opt/cuda-12.1:ro" \
    -v "${VATTN_HF_CACHE_HOST}:/root/.cache/huggingface" \
    -v "${VATTN_PIP_CACHE_HOST}:/root/.cache/pip" \
    -v "${VATTN_TORCH_CACHE_HOST}:/root/.cache/torch" \
    "${VATTN_IMAGE_NAME}"
