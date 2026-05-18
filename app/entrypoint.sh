#!/usr/bin/env bash
set -euo pipefail

# Wait for model artifacts to exist before starting the app
MODEL_DIR="/app/model"
MODEL_FILE="$MODEL_DIR/model.pkl"
TIMEOUT=${MODEL_WAIT_TIMEOUT:-300}
SLEEP_INTERVAL=2

echo "[entrypoint] waiting for model artifacts at $MODEL_DIR (timeout=${TIMEOUT}s)"

start_ts=$(date +%s)
while [ ! -f "$MODEL_FILE" ]; do
  now=$(date +%s)
  elapsed=$((now - start_ts))
  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "[entrypoint] timeout waiting for model file: $MODEL_FILE" >&2
    break
  fi
  sleep $SLEEP_INTERVAL
done

# Exec the CMD
exec "$@"
