"""
coding-agent-insights — Cursor adapter

Normalises raw Cursor hook events (JSON dicts from trace-hook.sh) into
NormalizedEvent objects for the core engine.
"""
import json
import os
import tempfile

from hooks.core import NormalizedEvent, log


BUFFER_PATH = os.environ.get(
    "CURSOR_TRACES_BUFFER",
    os.environ.get("TRACES_BUFFER", "/tmp/cursor-traces.jsonl"),
)

# ── Hook event → event_type mapping ──────────────────────────────────────────

HOOK_TO_EVENT_TYPE = {
    "sessionStart": "session_start",
    "beforeSubmitPrompt": "prompt",
    "afterAgentThought": "thinking",
    "afterAgentResponse": "response",
    "preCompact": "compaction",
    "stop": "session_end",
    "sessionEnd": "session_end",
    "postToolUse": "tool_use",
    "postToolUseFailure": "tool_error",
    "afterShellExecution": "shell",
    "afterMCPExecution": "mcp",
    "afterFileEdit": "file_edit",
    "subagentStop": "subagent",
}


class CursorAdapter:
    """Read Cursor hook events from the JSONL buffer and normalise them."""

    agent_type = "cursor"

    def read_events(self) -> list[NormalizedEvent]:
        """Read and drain the Cursor buffer file, returning NormalizedEvents."""
        lines = self._read_and_drain_buffer()
        if not lines:
            return []

        events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"Cursor: skipping malformed line: {e}")
                continue
            events.append(self._normalise(raw))
        return events

    # ── Buffer I/O (atomic drain) ─────────────────────────────────────────────

    def _read_and_drain_buffer(self) -> list[str]:
        """Atomically read and drain the buffer file.

        Uses rename-and-read to avoid a race where events appended between
        our read() and truncate() are silently lost.
        """
        if not os.path.exists(BUFFER_PATH):
            return []

        buf_dir = os.path.dirname(BUFFER_PATH) or "/tmp"
        drain_path = os.path.join(
            buf_dir,
            f".cursor-traces-drain-{os.getpid()}.jsonl",
        )

        try:
            os.rename(BUFFER_PATH, drain_path)
        except FileNotFoundError:
            return []
        except OSError as e:
            log(f"Cursor: rename failed, falling back to direct read: {e}")
            try:
                with open(BUFFER_PATH) as f:
                    lines = f.readlines()
                open(BUFFER_PATH, "w").close()
                return lines
            except Exception as e2:
                log(f"Cursor: fallback read failed: {e2}")
                return []

        try:
            with open(drain_path) as f:
                lines = f.readlines()
        finally:
            try:
                os.unlink(drain_path)
            except OSError:
                pass

        return lines

    # ── Event normalisation ───────────────────────────────────────────────────

    def _normalise(self, raw: dict) -> NormalizedEvent:
        """Convert a raw Cursor hook event dict to a NormalizedEvent."""
        hook = raw.get("hook_event_name", "unknown")
        event_type = HOOK_TO_EVENT_TYPE.get(hook, hook)

        event = NormalizedEvent(
            event_type=event_type,
            conversation_id=raw.get("conversation_id", ""),
            timestamp=float(raw.get("_timestamp", 0)),
            agent_type="cursor",
            user_id=str(raw.get("user_email", "")),
            model=str(raw.get("model", "")),
        )

        # Duration
        dur = raw.get("duration_ms") or raw.get("duration") or 0
        try:
            event.duration_ms = float(dur)
        except (ValueError, TypeError):
            event.duration_ms = 0

        # Name
        event.name = self._make_name(hook, raw)

        # Input / Output
        self._extract_io(event, hook, raw)

        # Error
        if hook == "postToolUseFailure" or raw.get("status") == "error":
            event.is_error = True
            event.error_message = str(raw.get("error_message", ""))

        # Extra attributes
        self._extract_attrs(event, hook, raw)

        return event

    def _make_name(self, hook: str, raw: dict) -> str:
        """Generate a span name from the hook event type."""
        if hook == "postToolUse":
            return f"tool:{raw.get('tool_name', 'unknown')}"
        if hook == "postToolUseFailure":
            return f"tool:{raw.get('tool_name', 'unknown')}.error"
        if hook == "afterShellExecution":
            return "shell"
        if hook == "afterMCPExecution":
            return f"mcp:{raw.get('tool_name', 'unknown')}"
        if hook == "afterFileEdit":
            fp = raw.get("file_path", "unknown")
            return f"edit:{os.path.basename(fp)}"
        if hook == "subagentStop":
            return f"subagent:{raw.get('subagent_type', 'unknown')}"
        # For standard types, the core will use the event_type default
        return ""

    def _extract_io(self, event: NormalizedEvent, hook: str, raw: dict) -> None:
        """Extract input/output values from a raw event."""
        if hook == "beforeSubmitPrompt":
            event.input_value = raw.get("prompt", "")
            if "attachments" in raw:
                event.attributes["attachments"] = raw["attachments"]

        elif hook == "afterAgentThought":
            event.output_value = raw.get("text", "")

        elif hook == "afterAgentResponse":
            event.output_value = raw.get("text", "")

        elif hook in ("postToolUse", "postToolUseFailure"):
            if "tool_input" in raw:
                event.input_value = json.dumps(raw["tool_input"]) if isinstance(raw["tool_input"], (dict, list)) else str(raw["tool_input"])
                event.input_mime_type = "application/json"
            if "tool_output" in raw:
                event.output_value = str(raw["tool_output"])

        elif hook == "afterShellExecution":
            event.input_value = str(raw.get("command", ""))
            event.output_value = str(raw.get("output", ""))

        elif hook == "afterMCPExecution":
            if "tool_input" in raw:
                event.input_value = json.dumps(raw["tool_input"]) if isinstance(raw["tool_input"], (dict, list)) else str(raw["tool_input"])
                event.input_mime_type = "application/json"
            if "result_json" in raw:
                event.output_value = str(raw["result_json"])
                event.output_mime_type = "application/json"

        elif hook == "afterFileEdit":
            event.input_value = raw.get("file_path", "")
            if "edits" in raw:
                event.output_value = json.dumps(raw["edits"])
                event.output_mime_type = "application/json"

        elif hook == "subagentStop":
            event.input_value = str(raw.get("task", ""))
            event.output_value = str(raw.get("summary", ""))

    def _extract_attrs(self, event: NormalizedEvent, hook: str, raw: dict) -> None:
        """Extract extra attributes from a raw event."""
        # Common attributes
        for k in ("conversation_id", "generation_id", "hook_event_name", "cursor_version"):
            if k in raw and raw[k] is not None:
                event.attributes[k] = str(raw[k])

        if hook == "sessionStart":
            for k in ("composer_mode", "is_background_agent"):
                if k in raw:
                    event.attributes[k] = str(raw[k])

        elif hook in ("postToolUse", "postToolUseFailure", "afterMCPExecution"):
            if "tool_name" in raw:
                event.attributes["tool.name"] = str(raw["tool_name"])
            if "duration" in raw:
                event.attributes["duration"] = str(raw["duration"])
            if hook == "postToolUseFailure":
                for k in ("failure_type",):
                    if k in raw:
                        event.attributes[k] = str(raw[k])

        elif hook == "afterFileEdit":
            if "file_path" in raw:
                event.attributes["file_path"] = raw["file_path"]

        elif hook == "preCompact":
            for k in ("context_tokens", "context_window_size", "context_usage_percent",
                       "message_count", "trigger"):
                if k in raw:
                    event.attributes[k] = str(raw[k])

        elif hook == "subagentStop":
            for k in ("subagent_type", "status", "duration_ms",
                       "tool_call_count", "message_count"):
                if k in raw:
                    event.attributes[k] = str(raw[k])

        elif hook in ("stop", "sessionEnd"):
            for k in ("status", "reason", "duration_ms"):
                if k in raw:
                    event.attributes[k] = str(raw[k])

        elif hook == "afterAgentThought":
            if "duration_ms" in raw:
                event.attributes["duration_ms"] = str(raw["duration_ms"])
