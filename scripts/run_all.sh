#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_all.py" "$@"
