#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

exec "${SCRIPT_DIR}/start-server.sh" \
    --model_name 01-ai/Yi-6B-200k \
    --model_tensor_parallel_degree 4 \
    --model_attention_backend fa_vattn \
    --model_load_format auto \
    --model_max_model_len 32768 \
    --gpu_memory_utilization 0.8 \
    --host 0.0.0.0 \
    --port 8000 \
    "$@"

