#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

VATTN_SERVER_OUTPUT_DIR=${VATTN_SERVER_OUTPUT_DIR:-/workspace/server-output/deepseek-v2-lite}
DEEPSEEK_V2_LITE_CUDA_VISIBLE_DEVICES=${DEEPSEEK_V2_LITE_CUDA_VISIBLE_DEVICES:-0,1,2,3}
DEEPSEEK_V2_LITE_DEFAULT_TP=${DEEPSEEK_V2_LITE_DEFAULT_TP:-4}
export VATTN_SERVER_OUTPUT_DIR

requested_tp="${DEEPSEEK_V2_LITE_DEFAULT_TP}"
requested_max_model_len=""
next_is_tp=0
next_is_max_model_len=0
for arg in "$@"; do
    if [[ "${next_is_tp}" == 1 ]]; then
        requested_tp="${arg}"
        next_is_tp=0
        continue
    fi
    if [[ "${next_is_max_model_len}" == 1 ]]; then
        requested_max_model_len="${arg}"
        next_is_max_model_len=0
        continue
    fi
    case "${arg}" in
        --model_tensor_parallel_degree)
            next_is_tp=1
            ;;
        --model_tensor_parallel_degree=*)
            requested_tp="${arg#*=}"
            ;;
        --model_max_model_len)
            next_is_max_model_len=1
            ;;
        --model_max_model_len=*)
            requested_max_model_len="${arg#*=}"
            ;;
    esac
done

if [[ -z "${requested_max_model_len}" ]]; then
    if [[ "${requested_tp}" -ge 4 ]]; then
        requested_max_model_len=32768
    else
        requested_max_model_len=128
    fi
fi

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
    --model_tensor_parallel_degree '"${requested_tp}"' \
    --model_attention_backend fa_vattn \
    --model_block_size 2097152 \
    --model_load_format auto \
    --model_max_model_len '"${requested_max_model_len}"' \
    --gpu_memory_utilization 0.85 \
    --replica_scheduler_max_batch_size 1 \
    --host 0.0.0.0 \
    --port 8000 \
    "$@"
' bash "${VATTN_SERVER_OUTPUT_DIR}" "$@"
