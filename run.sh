#!/usr/bin/env bash
# Launch the J-space visualizer.
#   ./run.sh              -> 127.0.0.1:8765
#   ./run.sh 9000         -> 127.0.0.1:9000
#   HOST=0.0.0.0 ./run.sh -> listen on all interfaces (LAN-visible)
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
PORT="${1:-8765}"
HOST="${HOST:-127.0.0.1}"
echo "J-space visualizer -> http://${HOST}:${PORT}"
exec python -m uvicorn jspace.serve:app --host "${HOST}" --port "${PORT}"
