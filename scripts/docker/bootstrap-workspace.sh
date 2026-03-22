#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

"${SCRIPT_DIR}/bootstrap-sarathi.sh"
"${SCRIPT_DIR}/bootstrap-pod-attn.sh"
"${SCRIPT_DIR}/bootstrap-vattention.sh"
