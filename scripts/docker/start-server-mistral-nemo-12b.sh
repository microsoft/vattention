#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VATTN_SERVER_OUTPUT_DIR=${VATTN_SERVER_OUTPUT_DIR:-/workspace/server-output/mistral-nemo-12b}
VATTN_MODEL_MAX_MODEL_LEN=${VATTN_MODEL_MAX_MODEL_LEN:-32768}
export VATTN_SERVER_OUTPUT_DIR
export VATTN_MODEL_MAX_MODEL_LEN

exec "${SCRIPT_DIR}/start-server.sh" \
    --model_name mistralai/Mistral-Nemo-Base-2407 \
    --model_tensor_parallel_degree 4 \
    --model_attention_backend fa_vattn \
    --model_load_format auto \
    --model_max_model_len "${VATTN_MODEL_MAX_MODEL_LEN}" \
    --gpu_memory_utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 \
    "$@"
