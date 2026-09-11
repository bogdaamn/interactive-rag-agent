#!/usr/bin/env bash
# Stop the interactive-rag-agent Telegram bot background process, and stop
# the local Ollama server too (freeing the CPU/RAM its loaded model was
# using).
# Intended to be invoked via the `stop interactive-rag` shell function
# (see the block appended to ~/.zshrc), but can also be run directly:
#   ./scripts/stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/.bot.pid"
OLLAMA_PID_FILE="$PROJECT_DIR/.ollama.pid"

cd "$PROJECT_DIR"

# Send TERM, wait up to 5s, then force-kill if it's still alive.
_stop_pid() {
  local pid="$1"
  local label="$2"

  kill "$pid" 2>/dev/null || true

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "$label did not stop gracefully, forcing kill (PID $pid)."
    kill -9 "$pid" 2>/dev/null || true
  fi
}

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    _stop_pid "$PID" "interactive-rag-agent bot"
    echo "interactive-rag-agent bot stopped."
  else
    echo "interactive-rag-agent bot is not running (stale PID file removed)."
  fi
  rm -f "$PID_FILE"
else
  echo "interactive-rag-agent bot is not running (no PID file)."
fi

# Stop Ollama regardless of whether this script started it. On machines
# running the Ollama.app menu-bar app, that app supervises "ollama serve"
# and respawns it the instant it dies — so the app itself must be quit
# FIRST, before its child server, or the server just comes right back.
OLLAMA_APP_PID="$(pgrep -f 'Ollama\.app/Contents/MacOS/Ollama' 2>/dev/null | head -n1 || true)"
if [[ -n "$OLLAMA_APP_PID" ]]; then
  _stop_pid "$OLLAMA_APP_PID" "Ollama.app"
fi

OLLAMA_PIDS=""
if [[ -f "$OLLAMA_PID_FILE" ]]; then
  OLLAMA_PIDS="$(cat "$OLLAMA_PID_FILE")"
fi
rm -f "$OLLAMA_PID_FILE"

SERVE_PIDS="$(pgrep -f 'ollama serve' 2>/dev/null || true)"
OLLAMA_PIDS="$(printf '%s\n%s\n' "$OLLAMA_PIDS" "$SERVE_PIDS" | grep -E '^[0-9]+$' | sort -u || true)"

if [[ -z "$OLLAMA_APP_PID" && -z "$OLLAMA_PIDS" ]]; then
  echo "Ollama is not running."
else
  for pid in $OLLAMA_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      _stop_pid "$pid" "Ollama server"
    fi
  done
  echo "Ollama stopped."
fi
