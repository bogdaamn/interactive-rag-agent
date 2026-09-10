#!/usr/bin/env bash
# Stop Ollama for local-rag-mcp, freeing the CPU/RAM its loaded model was
# using. There's no background local-rag process to stop separately -
# the assistant itself (main.py) is an interactive CLI you exit yourself
# (exit/quit/Ctrl+C); this script is only about the underlying model
# service, which otherwise keeps running after you leave the CLI.
#
# Stops Ollama regardless of who started it. On machines running the
# Ollama.app menu-bar app, that app supervises "ollama serve" and
# respawns it the instant it dies - so the app itself must be quit
# FIRST, before its child server, or the server just comes right back.
#
# Intended to be invoked via the `stop local-rag` shell function
# (see the block appended to ~/.zshrc), but can also be run directly:
#   ./scripts/stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OLLAMA_PID_FILE="$PROJECT_DIR/.ollama.pid"

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
