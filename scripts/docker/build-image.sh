#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

run_cmd docker build \
    -f "${REPO_ROOT}/docker/Dockerfile" \
    -t "${VATTN_IMAGE_NAME}" \
    "${REPO_ROOT}"

