#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

VATTN_SERVER_OUTPUT_DIR=${VATTN_SERVER_OUTPUT_DIR:-/tmp/vattention/${VATTN_CONTAINER_NAME}}
DEEPSEEK_V2_LITE_CUDA_VISIBLE_DEVICES=${DEEPSEEK_V2_LITE_CUDA_VISIBLE_DEVICES:-0,1}

if ! container_exists; then
    printf 'Container does not exist yet: %s\nRun scripts/docker/create-container.sh first.\n' "${VATTN_CONTAINER_NAME}" >&2
    exit 1
fi

if ! container_running; then
    run_cmd docker start "${VATTN_CONTAINER_NAME}"
fi

run_cmd docker exec \
    -it \
    -e "CUDA_VISIBLE_DEVICES=${DEEPSEEK_V2_LITE_CUDA_VISIBLE_DEVICES}" \
    "${VATTN_CONTAINER_NAME}" \
    bash -lc '
set -euo pipefail
output_dir="$1"
mkdir -p "$output_dir"
cd /workspace/sarathi-lean
shift
exec python -m sarathi.entrypoints.openai_server.api_server \
    --output_dir "$output_dir" \
    --model_name deepseek-ai/DeepSeek-V2-Lite \
    --model_tensor_parallel_degree 2 \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152 \
    --model_load_format auto \
    --model_max_model_len 128 \
    --gpu_memory_utilization 1.0 \
    --replica_scheduler_max_batch_size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    "$@"
' bash "${VATTN_SERVER_OUTPUT_DIR}" "$@"
