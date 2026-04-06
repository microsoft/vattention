#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/docker/common.sh"

SWEEP_PYTHON="${REPO_ROOT}/.venv-frag-sweep/bin/python"
PLOT_PYTHON="${REPO_ROOT}/.venv-londy/bin/python"
SWEEP_SCRIPT="${REPO_ROOT}/scripts/fragmentation_context_sweep.py"
PLOT_SCRIPT="${REPO_ROOT}/scripts/plotting/plot_context_vs_fragmentation.py"

MODEL_KEY=""
CONTEXT_LENGTHS=""
PORT=8000
WAIT_TIMEOUT=180
SHUTDOWN_TIMEOUT=60
METRICS_SETTLE_SECONDS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-key)
            MODEL_KEY="$2"
            shift 2
            ;;
        --context-lengths)
            CONTEXT_LENGTHS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --wait-timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        --shutdown-timeout)
            SHUTDOWN_TIMEOUT="$2"
            shift 2
            ;;
        --metrics-settle-seconds)
            METRICS_SETTLE_SECONDS="$2"
            shift 2
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${MODEL_KEY}" ]]; then
    printf 'Usage: %s --model-key {qwen-14b|mistral-nemo-12b|mistral-nemo-12b-mla|llama-3-8b|deepseek-v2-lite} [--context-lengths CSV] [--port N]\n' "$0" >&2
    exit 1
fi

case "${MODEL_KEY}" in
    qwen-14b)
        WRAPPER="${REPO_ROOT}/scripts/docker/start-server-qwen14b.sh"
        MODEL_NAME="Qwen/Qwen-14B"
        MODEL_SLUG="qwen-14b"
        PLOT_TITLE="Qwen-14B (MHA): Context Length vs Fragmentation"
        ;;
    mistral-nemo-12b)
        WRAPPER="${REPO_ROOT}/scripts/docker/start-server-mistral-nemo-12b.sh"
        MODEL_NAME="mistralai/Mistral-Nemo-Base-2407"
        MODEL_SLUG="mistral-nemo-12b"
        PLOT_TITLE="Mistral-Nemo-12B (GQA): Context Length vs Fragmentation"
        ;;
    mistral-nemo-12b-mla)
        WRAPPER="${REPO_ROOT}/scripts/docker/start-server-mistral-nemo-12b-mla.sh"
        MODEL_NAME="mistralai/Mistral-Nemo-Base-2407"
        MODEL_SLUG="mistral-nemo-12b-mla"
        PLOT_TITLE="Mistral-Nemo-12B (Synthetic MLA): Context Length vs Fragmentation"
        ;;
    llama-3-8b)
        WRAPPER="${REPO_ROOT}/scripts/docker/start-server-llama3-8b.sh"
        MODEL_NAME="meta-llama/Meta-Llama-3-8B"
        MODEL_SLUG="llama-3-8b"
        PLOT_TITLE="Llama-3-8B (GQA): Context Length vs Fragmentation"
        ;;
    deepseek-v2-lite)
        WRAPPER="${REPO_ROOT}/scripts/docker/start-server-deepseek-v2-lite.sh"
        MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"
        MODEL_SLUG="deepseek-v2-lite"
        PLOT_TITLE="DeepSeek-V2-Lite (MLA): Context Length vs Fragmentation"
        ;;
    *)
        printf 'Unsupported model key: %s\n' "${MODEL_KEY}" >&2
        exit 1
        ;;
esac

for required in "${SWEEP_PYTHON}" "${PLOT_PYTHON}" "${SWEEP_SCRIPT}" "${PLOT_SCRIPT}" "${WRAPPER}"; do
    if [[ ! -x "${required}" && ! -f "${required}" ]]; then
        printf 'Missing required path: %s\n' "${required}" >&2
        exit 1
    fi
done

SERVER_OUTPUT_HOST="${REPO_ROOT}/server-output/${MODEL_SLUG}"
SERVER_OUTPUT_CONTAINER="/workspace/server-output/${MODEL_SLUG}"
SERVER_PLOTS_DIR="${REPO_ROOT}/server_plots/${MODEL_SLUG}"
SERVER_LOG="${SERVER_PLOTS_DIR}/server.log"
PLOT_PATH="${SERVER_PLOTS_DIR}/context_vs_fragmentation.png"
SUMMARY_PATH="${SERVER_PLOTS_DIR}/context_vs_fragmentation_summary.csv"
METRICS_PATH="${SERVER_OUTPUT_HOST}/sequence_metrics.csv"
BENCHMARK_CONFIG_PATH="${SERVER_OUTPUT_HOST}/benchmark_config.yml"
export MPLCONFIGDIR=/tmp/mplconfig

mkdir -p "${SERVER_PLOTS_DIR}"

ensure_container_running
docker exec "${VATTN_CONTAINER_NAME}" bash -lc "pkill -SIGINT -f 'python -m sarathi.entrypoints.openai_server.api_server'" >/dev/null 2>&1 || true
sleep 2
docker exec "${VATTN_CONTAINER_NAME}" bash -lc "pkill -SIGTERM -f 'python -m sarathi.entrypoints.openai_server.api_server'" >/dev/null 2>&1 || true
docker exec "${VATTN_CONTAINER_NAME}" bash -lc 'output_dir="$1"; rm -rf "$output_dir"; mkdir -p "$output_dir"' bash "${SERVER_OUTPUT_CONTAINER}"

server_pid=""

cleanup() {
    set +e
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        docker exec "${VATTN_CONTAINER_NAME}" bash -lc "pkill -SIGINT -f 'python -m sarathi.entrypoints.openai_server.api_server'" >/dev/null 2>&1 || true
        wait "${server_pid}" >/dev/null 2>&1 || true
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill "${server_pid}" >/dev/null 2>&1 || true
        fi
    fi
}

trap cleanup EXIT

printf 'Starting %s server...\n' "${MODEL_NAME}"
"${WRAPPER}" --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!

printf 'Waiting for server readiness on port %s...\n' "${PORT}"
ready=0
for ((i=0; i<WAIT_TIMEOUT; i++)); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "${ready}" != "1" ]]; then
    printf 'Server did not become ready within %s seconds.\n' "${WAIT_TIMEOUT}" >&2
    exit 1
fi

models_response=$(curl -fsS "http://127.0.0.1:${PORT}/v1/models")
if [[ "${models_response}" != *"${MODEL_NAME}"* ]]; then
    printf 'Ready endpoint does not match expected model.\nExpected: %s\nResponse: %s\n' "${MODEL_NAME}" "${models_response}" >&2
    exit 1
fi

printf 'Running fragmentation sweep for %s...\n' "${MODEL_NAME}"
sweep_cmd=("${SWEEP_PYTHON}" "${SWEEP_SCRIPT}" --model "${MODEL_NAME}" --fail-fast)
if [[ -n "${CONTEXT_LENGTHS}" ]]; then
    sweep_cmd+=(--context-lengths "${CONTEXT_LENGTHS}")
fi
"${sweep_cmd[@]}"

printf 'Stopping server gracefully...\n'
docker exec "${VATTN_CONTAINER_NAME}" bash -lc "pkill -SIGINT -f 'python -m sarathi.entrypoints.openai_server.api_server'" >/dev/null 2>&1 || true

metrics_seen=0
for ((i=0; i<SHUTDOWN_TIMEOUT; i++)); do
    if [[ -f "${METRICS_PATH}" ]]; then
        metrics_seen=1
        break
    fi
    sleep 1
done

if [[ "${metrics_seen}" != "1" ]]; then
    printf 'sequence_metrics.csv did not appear within %s seconds of shutdown; sending SIGTERM fallback.\n' "${SHUTDOWN_TIMEOUT}"
    docker exec "${VATTN_CONTAINER_NAME}" bash -lc "pkill -SIGTERM -f 'python -m sarathi.entrypoints.openai_server.api_server'" >/dev/null 2>&1 || true
else
    printf 'Detected metrics file; waiting %s seconds for writes to settle...\n' "${METRICS_SETTLE_SECONDS}"
    sleep "${METRICS_SETTLE_SECONDS}"
fi

shutdown_complete=0
for ((i=0; i<SHUTDOWN_TIMEOUT; i++)); do
    if ! docker exec "${VATTN_CONTAINER_NAME}" bash -lc "ps -ef | grep -F 'sarathi.entrypoints.openai_server.api_server' | grep -v grep" >/dev/null 2>&1; then
        shutdown_complete=1
        break
    fi
    sleep 1
done

if [[ "${shutdown_complete}" != "1" ]]; then
    printf 'Server process still alive after settle window; sending SIGTERM fallback.\n'
    docker exec "${VATTN_CONTAINER_NAME}" bash -lc "pkill -SIGTERM -f 'python -m sarathi.entrypoints.openai_server.api_server'" >/dev/null 2>&1 || true
fi

wait "${server_pid}" >/dev/null 2>&1 || true
if kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" >/dev/null 2>&1 || true
fi
server_pid=""

if [[ ! -f "${METRICS_PATH}" ]]; then
    printf 'Expected metrics file was not written: %s\n' "${METRICS_PATH}" >&2
    exit 1
fi

printf 'Plotting results...\n'
"${PLOT_PYTHON}" "${PLOT_SCRIPT}" \
    --input "${METRICS_PATH}" \
    --out-plot "${PLOT_PATH}" \
    --out-summary "${SUMMARY_PATH}" \
    --title "${PLOT_TITLE}" \
    --bins 16

printf '\nPipeline complete.\n'
printf 'Benchmark config: %s\n' "${BENCHMARK_CONFIG_PATH}"
printf 'Metrics: %s\n' "${METRICS_PATH}"
printf 'Plot: %s\n' "${PLOT_PATH}"
printf 'Summary: %s\n' "${SUMMARY_PATH}"
printf 'Server log: %s\n' "${SERVER_LOG}"
