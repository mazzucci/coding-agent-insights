#!/bin/bash
# coding-agent-insights — bash hook script
# Appends hook event JSON to a buffer file. Triggers flush on stop/sessionEnd.
# This is the hot path (~5ms per invocation). The flush script (Python) handles
# the heavy lifting of converting events to Phoenix spans.

ENV_FILE="$(dirname "$0")/.coding-agent-insights.env"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

BUFFER="${CURSOR_TRACES_BUFFER:-/tmp/cursor-traces.jsonl}"
FLUSH_SCRIPT="$(dirname "$0")/flush.py"

INPUT=$(cat)

echo "$INPUT" >> "$BUFFER"

if echo "$INPUT" | grep -qE '"hook_event_name"\s*:\s*"(stop|sessionEnd)"'; then
    uv run "$FLUSH_SCRIPT" &
fi

exit 0
