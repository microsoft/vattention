#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
VENV_DIR="${REPO_ROOT}/.venv-londy"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-context-fragmentation.txt"

if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        uv venv "${VENV_DIR}"
    fi
    uv pip install --python "${VENV_DIR}/bin/python" -r "${REQUIREMENTS_FILE}"
else
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        python3 -m venv "${VENV_DIR}"
    fi
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"
fi

printf 'Created plotting environment at %s\n' "${VENV_DIR}"
printf 'Activate with:\n  source %s/bin/activate\n' "${VENV_DIR}"
printf 'Run the plotter with:\n  python %s/scripts/plotting/plot_context_vs_fragmentation.py --help\n' "${REPO_ROOT}"
