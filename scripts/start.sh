#!/usr/bin/env bash
# Set up local-rag-mcp (venv, Ollama, model, index) and launch the
# interactive assistant. Unlike a headless daemon, main.py reads
# questions from stdin, so this runs it in the FOREGROUND — it hands you
# straight into the "❓ Question:" prompt and blocks until you type
# exit/quit or press Ctrl+C. Ollama, once started, keeps running in the
# background after that; use `stop local-rag` to bring it down.
#
# Intended to be invoked via the `start local-rag` shell function
# (see the block appended to ~/.zshrc), but can also be run directly:
#   ./scripts/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
OLLAMA_PID_FILE="$PROJECT_DIR/.ollama.pid"
OLLAMA_LOG_FILE="$PROJECT_DIR/ollama.log"
OLLAMA_URL="http://localhost:11434"

cd "$PROJECT_DIR"

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON="$VENV_DIR/bin/python"
else
  echo "venv not found - creating it (first run only)..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install -q --upgrade pip
  "$VENV_DIR/bin/pip" install -q -r "$PROJECT_DIR/src/requirements.txt"
  PYTHON="$VENV_DIR/bin/python"
fi

OLLAMA_MODEL="$("$PYTHON" -c "import sys; sys.path.insert(0, '$PROJECT_DIR/src'); from config import OLLAMA_MODEL; print(OLLAMA_MODEL)")"

if curl -s -o /dev/null -m 2 "$OLLAMA_URL"; then
  echo "Ollama is already running at $OLLAMA_URL."
elif ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama is not installed. Install it first: https://ollama.ai" >&2
  exit 1
else
  echo "Starting Ollama server..."
  nohup ollama serve >> "$OLLAMA_LOG_FILE" 2>&1 &
  echo "$!" > "$OLLAMA_PID_FILE"
  disown

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s -o /dev/null -m 2 "$OLLAMA_URL"; then
      break
    fi
    sleep 1
  done

  if ! curl -s -o /dev/null -m 2 "$OLLAMA_URL"; then
    echo "Error: Ollama did not become reachable at $OLLAMA_URL within 10s. Last log lines:" >&2
    tail -n 15 "$OLLAMA_LOG_FILE" >&2 2>/dev/null || true
    exit 1
  fi
fi

if ! ollama list | awk '{print $1}' | grep -qx "$OLLAMA_MODEL"; then
  echo "Pulling model $OLLAMA_MODEL (configured in src/config.py)..."
  ollama pull "$OLLAMA_MODEL"
fi

if [[ ! -f "$PROJECT_DIR/src/index.faiss" || ! -f "$PROJECT_DIR/src/chunks.pkl" || ! -f "$PROJECT_DIR/src/fts.db" ]]; then
  if [[ -n "$(find "$PROJECT_DIR/src/docs" -maxdepth 1 -type f 2>/dev/null)" ]]; then
    echo "Building index from src/docs/..."
    (cd "$PROJECT_DIR/src" && "$PYTHON" main.py build-index)
  else
    echo "No index found and src/docs/ is empty - skipping build-index (add documents first)."
  fi
fi

echo "Launching assistant (type exit, quit, or Ctrl+C to leave the CLI)..."
cd "$PROJECT_DIR/src"
exec "$PYTHON" main.py
