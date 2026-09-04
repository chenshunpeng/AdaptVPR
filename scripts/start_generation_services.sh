#!/bin/bash
set -euo pipefail

ADAPTVPR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR=${ADAPTVPR_SERVICE_LOG_DIR:-$ADAPTVPR_ROOT/tmp/logs}
TMP_DIR=${ADAPTVPR_TMP_DIR:-$ADAPTVPR_ROOT/tmp}
HEALTHCHECK_PYTHON=${ADAPTVPR_HEALTHCHECK_PYTHON:-python}
ICLIGHT_ROOT=${ICLIGHT_ROOT:-${ICLIGHT_WORKDIR:-/path/to/IC-Light}}
ICLIGHT_PYTHON=${ICLIGHT_PYTHON:-python}
LIGHTX2V_ROOT=${LIGHTX2V_ROOT:-${LIGHTX2V_WORKDIR:-/path/to/LightX2V}}
LIGHTX2V_PYTHON=${LIGHTX2V_PYTHON:-python}
ICLIGHT_ADAPTER=$ADAPTVPR_ROOT/adapters/iclight_sd15_fc.py
LIGHTX2V_ADAPTER=$ADAPTVPR_ROOT/adapters/lightx2v_qwen_image_edit.py

mkdir -p "$LOG_DIR" "$TMP_DIR"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export ADAPTVPR_DISABLE_MOCK=${ADAPTVPR_DISABLE_MOCK:-1}
export ICLIGHT_ROOT LIGHTX2V_ROOT

is_port_open() {
  local port="$1"
  "$HEALTHCHECK_PYTHON" - "$port" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

start_service() {
  local name="$1"
  local port="$2"
  local workdir="$3"
  local python_bin="$4"
  local entrypoint="$5"
  local log_file="$LOG_DIR/${name}_${port}.log"
  local cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"
  local port_name

  if [ ! -d "$workdir" ]; then
    echo "[$name] missing upstream Git checkout: $workdir" >&2
    return 1
  fi
  if [ ! -f "$entrypoint" ]; then
    echo "[$name] missing AdaptVPR adapter: $entrypoint" >&2
    return 1
  fi
  if [ "$name" = "iclight" ]; then
    port_name=ICLIGHT_PORT
  else
    port_name=LIGHTX2V_PORT
  fi

  if [ "${ADAPTVPR_FORCE_RESTART:-0}" != "1" ] && is_port_open "$port"; then
    echo "[$name] port $port already open"
    return 0
  fi
  if [ "${ADAPTVPR_FORCE_RESTART:-0}" = "1" ] && is_port_open "$port"; then
    echo "[$name] stopping existing process on port $port"
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    sleep 2
  fi

  echo "[$name] starting pinned adapter on port $port"
  if [ "${ADAPTVPR_DISABLE_TMUX:-0}" != "1" ] && command -v tmux >/dev/null 2>&1; then
    local session="adaptvpr_${name}_${port}"
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
    fi
    tmux new-session -d -s "$session" \
      "cd '$workdir'; export CUDA_VISIBLE_DEVICES='$cuda_devices'; export $port_name='$port'; export PLATFORM=cuda; export SKIP_PLATFORM_CHECK=True; export PYTHONPATH='$workdir':\${PYTHONPATH:-}; export ICLIGHT_OUTPUT_DIR='$TMP_DIR/service_outputs/iclight'; export LIGHTX2V_OUTPUT_DIR='$TMP_DIR/service_outputs/lightx2v'; exec '$python_bin' '$entrypoint' >>'$log_file' 2>&1"
    echo "[$name] tmux=$session log=$log_file"
  else
    nohup bash -c '
      cd "$1"
      export CUDA_VISIBLE_DEVICES="$3"
      export "$5=$6"
      export PLATFORM=cuda
      export SKIP_PLATFORM_CHECK=True
      export PYTHONPATH="$1:${PYTHONPATH:-}"
      export ICLIGHT_OUTPUT_DIR="$7/service_outputs/iclight"
      export LIGHTX2V_OUTPUT_DIR="$7/service_outputs/lightx2v"
      exec "$2" "$4"
    ' _ "$workdir" "$python_bin" "$cuda_devices" "$entrypoint" "$port_name" "$port" "$TMP_DIR" \
      >>"$log_file" 2>&1 &
    echo "[$name] pid=$! log=$log_file"
  fi
}

start_service iclight "${ICLIGHT_PORT:-8002}" "$ICLIGHT_ROOT" "$ICLIGHT_PYTHON" "$ICLIGHT_ADAPTER"
start_service lightx2v "${LIGHTX2V_PORT:-8001}" "$LIGHTX2V_ROOT" "$LIGHTX2V_PYTHON" "$LIGHTX2V_ADAPTER"

echo "Generation service startup requested. Check /health until model_loaded and generator_ready are true."
