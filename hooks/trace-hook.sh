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

# Inject a high-resolution timestamp so flush.py can order spans
# correctly even when multiple hook events fire within the same
# wall-clock millisecond.  `date +%s.%N` gives nanosecond precision
# on Linux; macOS `date` only gives seconds, but GNU coreutils
# (gdate) or python can fill in.  We try the fast path first.
if command -v gdate >/dev/null 2>&1; then
    TS=$(gdate +%s.%N)
elif date +%s.%N 2>/dev/null | grep -q '\.'; then
    TS=$(date +%s.%N)
else
    TS=$(python3 -c 'import time; print(f"{time.time():.9f}")' 2>/dev/null || date +%s)
fi

# Append the timestamp field to the JSON object before the closing brace.
# This avoids pulling in jq as a dependency.
TAGGED=$(echo "$INPUT" | sed "s/}$/,\"_timestamp\":$TS}/")

echo "$TAGGED" >> "$BUFFER"

if echo "$INPUT" | grep -qE '"hook_event_name"\s*:\s*"(stop|sessionEnd)"'; then
    uv run "$FLUSH_SCRIPT" &
fi

exit 0
