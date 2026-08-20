#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DIR/.." && pwd)"
export LD_LIBRARY_PATH="$DIR/.qt-libs:${LD_LIBRARY_PATH:-}"
exec "$REPO/.venv/bin/python" "$DIR/desktop.py"
