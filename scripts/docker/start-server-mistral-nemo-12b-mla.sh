#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

VATTN_SERVER_OUTPUT_DIR=${VATTN_SERVER_OUTPUT_DIR:-/workspace/server-output/mistral-nemo-12b-mla}
VATTN_MODEL_MAX_MODEL_LEN=${VATTN_MODEL_MAX_MODEL_LEN:-32768}
VATTN_MISTRAL_MLA_KV_LORA_RANK=${VATTN_MISTRAL_MLA_KV_LORA_RANK:-128}
VATTN_MISTRAL_MLA_QK_ROPE_HEAD_DIM=${VATTN_MISTRAL_MLA_QK_ROPE_HEAD_DIM:-64}
VATTN_MISTRAL_MLA_QK_NOPE_HEAD_DIM=${VATTN_MISTRAL_MLA_QK_NOPE_HEAD_DIM:-64}
VATTN_MISTRAL_MLA_V_HEAD_DIM=${VATTN_MISTRAL_MLA_V_HEAD_DIM:-128}
export VATTN_SERVER_OUTPUT_DIR

if ! container_exists; then
    printf 'Container does not exist yet: %s\nRun scripts/docker/create-container.sh first.\n' "${VATTN_CONTAINER_NAME}" >&2
    exit 1
fi

if ! container_running; then
    run_cmd docker start "${VATTN_CONTAINER_NAME}"
fi

readarray -t exec_args < <(docker_exec_args)

run_cmd docker exec \
    "${exec_args[@]}" \
    -e "VATTN_ENABLE_MISTRAL_MLA_CONVERSION=1" \
    -e "VATTN_MISTRAL_MLA_KV_LORA_RANK=${VATTN_MISTRAL_MLA_KV_LORA_RANK}" \
    -e "VATTN_MISTRAL_MLA_QK_ROPE_HEAD_DIM=${VATTN_MISTRAL_MLA_QK_ROPE_HEAD_DIM}" \
    -e "VATTN_MISTRAL_MLA_QK_NOPE_HEAD_DIM=${VATTN_MISTRAL_MLA_QK_NOPE_HEAD_DIM}" \
    -e "VATTN_MISTRAL_MLA_V_HEAD_DIM=${VATTN_MISTRAL_MLA_V_HEAD_DIM}" \
    "${VATTN_CONTAINER_NAME}" \
    bash -lc '
set -euo pipefail
output_dir="$1"
mkdir -p "$output_dir"
cd /workspace/sarathi-lean
shift
exec python -m sarathi.entrypoints.openai_server.api_server \
    --output_dir "$output_dir" \
    --model_name mistralai/Mistral-Nemo-Base-2407 \
    --model_tensor_parallel_degree 4 \
    --model_attention_backend fa_vattn \
    --model_load_format auto \
    --model_max_model_len '"${VATTN_MODEL_MAX_MODEL_LEN}"' \
    --gpu_memory_utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 \
    "$@"
' bash "${VATTN_SERVER_OUTPUT_DIR}" "$@"
