#!/usr/bin/env bash
# Launch the J-space visualizer.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
PORT="${1:-8765}"
echo "J-space visualizer -> http://127.0.0.1:${PORT}"
exec python -m uvicorn jspace.serve:app --host 127.0.0.1 --port "${PORT}"
