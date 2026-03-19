#!/bin/bash
# coding-agent-insights — Claude Code hook script
#
# Fires on Claude Code's "Stop" hook (after each agent turn).
# Reads the transcript_path from the hook's JSON stdin and triggers
# flush.py to parse the JSONL transcript and send spans to Phoenix.
#
# Claude Code hooks pass a JSON object on stdin with:
#   session_id, hook_event_name, transcript_path, ...
#
# Install by adding to ~/.claude/settings.json:
#   {
#     "hooks": {
#       "Stop": [{ "command": "bash /path/to/claude-code-trace-hook.sh" }]
#     }
#   }

ENV_FILE="$(dirname "$0")/.coding-agent-insights.env"
# shellcheck source=/dev/null
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

FLUSH_SCRIPT="$(dirname "$0")/flush.py"

# Read the hook context from stdin
INPUT=$(cat)

# Extract transcript_path from the JSON input
# Uses python for reliable JSON parsing (no jq dependency)
TRANSCRIPT_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('transcript_path', ''))
except Exception:
    pass
" 2>/dev/null)

if [ -z "$TRANSCRIPT_PATH" ]; then
    # Fallback: try to find the session transcript from session_id
    SESSION_ID=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('session_id', ''))
except Exception:
    pass
" 2>/dev/null)

    if [ -n "$SESSION_ID" ]; then
        # Search for the transcript in the default Claude projects directory
        CLAUDE_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"
        TRANSCRIPT_PATH=$(find "$CLAUDE_DIR" -name "${SESSION_ID}.jsonl" -type f 2>/dev/null | head -1)
    fi
fi

if [ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ]; then
    exit 0  # No transcript to process — exit silently
fi

# Run flush.py with the Claude Code adapter in the background
uv run "$FLUSH_SCRIPT" --agent claude_code --transcript "$TRANSCRIPT_PATH" &

exit 0
