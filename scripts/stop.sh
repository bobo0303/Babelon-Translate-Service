#!/bin/bash

# ============================================
# Babelon Translation Service - Stop
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NOTIFY_SCRIPT="$SCRIPT_DIR/notify.sh"

# Load notification config
if [ -f "$PROJECT_DIR/.env.notify" ]; then
    source "$PROJECT_DIR/.env.notify"
fi

# Change to project directory
cd "$PROJECT_DIR"

echo "Stopping Babelon Translation Service..."

# Send shutdown notification first
if [ "$NOTIFY_ON_STOP" = "true" ] && [ -f "$NOTIFY_SCRIPT" ]; then
    bash "$NOTIFY_SCRIPT" shutdown
fi

docker compose down
echo "Service stopped. Data in ./audio, ./logs, ./models is preserved."
echo "To start again: ./scripts/start.sh"
