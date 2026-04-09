#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ensure_container_running

run_cmd docker exec -it "${VATTN_CONTAINER_NAME}" bash
