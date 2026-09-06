#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VATTN_SERVER_OUTPUT_DIR=${VATTN_SERVER_OUTPUT_DIR:-/workspace/server-output/qwen-14b}
export VATTN_SERVER_OUTPUT_DIR

exec "${SCRIPT_DIR}/start-server.sh" \
    --model_name Qwen/Qwen-14B \
    --model_tensor_parallel_degree 4 \
    --model_attention_backend fa_vattn \
    --model_load_format auto \
    --model_max_model_len 8192 \
    --gpu_memory_utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 \
    "$@"
