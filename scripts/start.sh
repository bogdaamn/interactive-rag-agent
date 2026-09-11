#!/usr/bin/env bash
# Start the interactive-rag-agent Telegram bot (src/telegram_bot/main.py) as a
# background process, along with the local Ollama server (started if not
# already reachable) and a warm-up request so the configured model is loaded
# into memory before the first real Telegram message arrives.
#
# Follows the same start/stop convention as the sibling telegram-bot and
# local-rag-mcp projects. The bot's package imports (`telegram_bot.*`,
# `userdocs.*`) resolve relative to src/, so this runs it via
# `python -m telegram_bot.main` with PYTHONPATH=src rather than executing
# src/telegram_bot/main.py directly (which fails with ModuleNotFoundError).
#
# Intended to be invoked via the `start interactive-rag` shell function
# (see the block appended to ~/.zshrc), but can also be run directly:
#   ./scripts/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/.bot.pid"
LOG_FILE="$PROJECT_DIR/bot.log"
OLLAMA_PID_FILE="$PROJECT_DIR/.ollama.pid"
OLLAMA_LOG_FILE="$PROJECT_DIR/ollama.log"

cd "$PROJECT_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "interactive-rag-agent bot is already running (PID $(cat "$PID_FILE"))."
  exit 0
fi

if [[ ! -f src/telegram_bot/main.py ]]; then
  echo "Error: src/telegram_bot/main.py not found in $PROJECT_DIR." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Error: .env not found. Run: cp .env.example .env, then set TELEGRAM_BOT_TOKEN." >&2
  exit 1
fi

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  echo "Error: .venv not found. Run: python3 -m venv .venv && .venv/bin/pip install -r src/requirements.txt" >&2
  exit 1
fi

OLLAMA_BASE_URL="$(grep -E '^OLLAMA_BASE_URL=' .env 2>/dev/null | cut -d= -f2- || true)"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="$(grep -E '^OLLAMA_MODEL=' .env 2>/dev/null | cut -d= -f2- || true)"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"

if curl -s -o /dev/null -m 2 "$OLLAMA_BASE_URL"; then
  echo "Ollama is already running at $OLLAMA_BASE_URL."
elif ! command -v ollama >/dev/null 2>&1; then
  echo "Warning: ollama is not installed or not on PATH — the bot will start, but replies will fail until Ollama is running." >&2
else
  echo "Starting Ollama server..."
  nohup ollama serve >> "$OLLAMA_LOG_FILE" 2>&1 &
  echo "$!" > "$OLLAMA_PID_FILE"
  disown

  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -s -o /dev/null -m 2 "$OLLAMA_BASE_URL"; then
      break
    fi
    sleep 1
  done

  if ! curl -s -o /dev/null -m 2 "$OLLAMA_BASE_URL"; then
    echo "Warning: Ollama did not become reachable at $OLLAMA_BASE_URL within 10s — the bot will start, but replies will fail until it is. Last log lines:" >&2
    tail -n 15 "$OLLAMA_LOG_FILE" >&2 2>/dev/null || true
  fi
fi

if curl -s -o /dev/null -m 2 "$OLLAMA_BASE_URL"; then
  echo "Warming up model '$OLLAMA_MODEL' (loading into memory)..."
  WARMUP_STATUS="$(curl -s -o /dev/null -w '%{http_code}' -m 60 -X POST "$OLLAMA_BASE_URL/api/chat" \
    -d "{\"model\": \"$OLLAMA_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"stream\": false}")"
  if [[ "$WARMUP_STATUS" != "200" ]]; then
    echo "Warning: model warm-up request returned HTTP $WARMUP_STATUS — check that '$OLLAMA_MODEL' has been pulled (ollama pull $OLLAMA_MODEL)." >&2
  fi
fi

PYTHONPATH="$PROJECT_DIR/src" nohup "$PYTHON" -m telegram_bot.main >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
disown

# Give it a moment to crash on startup (bad token, missing deps, etc.)
# before declaring success, instead of reporting "started" unconditionally.
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "Error: interactive-rag-agent bot exited immediately after starting. Last log lines:" >&2
  tail -n 15 "$LOG_FILE" >&2
  exit 1
fi

echo "interactive-rag-agent bot started (PID $PID). Logs: $LOG_FILE"
