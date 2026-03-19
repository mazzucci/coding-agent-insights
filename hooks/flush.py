# /// script
# requires-python = ">=3.10"
# dependencies = ["arize-phoenix-client>=2.0.0"]
# ///
"""
coding-agent-insights — flush entrypoint

Reads buffered hook events, converts them to Phoenix spans, and sends them.
Supports multiple coding agents through the adapter pattern.

Usage:
    # Cursor (default — reads from JSONL buffer)
    uv run flush.py

    # Cursor (explicit)
    uv run flush.py --agent cursor

    # Claude Code (reads transcript from path)
    uv run flush.py --agent claude_code --transcript /path/to/session.jsonl

    # Claude Code (reads transcript path from stdin, as used by hooks)
    echo '{"transcript_path":"/path/to/session.jsonl"}' | uv run flush.py --agent claude_code
"""
import argparse
import json
import os
import sys

# Allow imports when run via `uv run` from the hooks directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.core import log, process_and_send
from hooks.adapters.cursor import CursorAdapter
from hooks.adapters.claude_code import ClaudeCodeAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="coding-agent-insights — flush traces to Phoenix"
    )
    parser.add_argument(
        "--agent",
        choices=["cursor", "claude_code"],
        default=os.environ.get("AGENT_TYPE", "cursor"),
        help="Coding agent type (default: cursor, or AGENT_TYPE env var)",
    )
    parser.add_argument(
        "--transcript",
        default=os.environ.get("CLAUDE_TRANSCRIPT_PATH", ""),
        help="Path to Claude Code transcript JSONL (Claude Code only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.agent == "cursor":
        adapter = CursorAdapter()
        events = adapter.read_events()
    elif args.agent == "claude_code":
        adapter = ClaudeCodeAdapter()
        transcript_path = args.transcript

        # If no transcript path provided, try reading from stdin
        # (Claude Code hooks pass context as JSON on stdin)
        if not transcript_path and not sys.stdin.isatty():
            try:
                stdin_data = sys.stdin.read()
                if stdin_data.strip():
                    hook_context = json.loads(stdin_data)
                    transcript_path = hook_context.get("transcript_path", "")
            except (json.JSONDecodeError, Exception) as e:
                log(f"Failed to read transcript path from stdin: {e}")

        if not transcript_path:
            log("Claude Code: no transcript path — nothing to flush")
            return

        events = adapter.read_events(transcript_path=transcript_path)
    else:
        log(f"Unknown agent type: {args.agent}")
        return

    if not events:
        log(f"No events from {args.agent} adapter")
        return

    log(f"Flushing {len(events)} events from {args.agent} to Phoenix")
    process_and_send(events)


if __name__ == "__main__":
    main()
