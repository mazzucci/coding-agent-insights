# /// script
# requires-python = ">=3.10"
# dependencies = ["arize-phoenix-client>=2.0.0"]
# ///
"""
coding-agent-insights — flush script
Reads buffered hook events from a JSONL file, converts them to Phoenix spans,
sends them via the Phoenix client SDK, and truncates the buffer.

Only runs on stop/sessionEnd events (called by trace-hook.sh).
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone

BUFFER_PATH = os.environ.get("CURSOR_TRACES_BUFFER", "/tmp/cursor-traces.jsonl")
PHOENIX_HOST = os.environ.get("PHOENIX_HOST", "http://localhost:6006")
PHOENIX_PROJECT = os.environ.get("PHOENIX_PROJECT", "cursor")
SKIP_FIELDS = set(
    f.strip()
    for f in os.environ.get("CURSOR_TRACES_SKIP", "").split(",")
    if f.strip()
)
DEBUG = os.environ.get("CURSOR_TRACES_DEBUG", "").lower() in ("1", "true", "yes")
LOG_PATH = "/tmp/cursor-traces.log"


def log(msg: str) -> None:
    if not DEBUG:
        return
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")
    except Exception:
        pass


def redact(event: dict) -> dict:
    if not SKIP_FIELDS:
        return event
    cleaned = {}
    for k, v in event.items():
        if k in SKIP_FIELDS:
            cleaned[k] = "[redacted]"
        elif isinstance(v, dict):
            cleaned[k] = redact(v)
        else:
            cleaned[k] = v
    return cleaned


def make_trace_id(conversation_id: str, turn_index: int = 0) -> str:
    key = f"{conversation_id}:turn:{turn_index}"
    return uuid.uuid5(uuid.NAMESPACE_URL, key).hex


def make_span_id() -> str:
    return uuid.uuid4().hex[:16]


def event_to_span_name(event: dict) -> str:
    hook = event.get("hook_event_name", "unknown")
    static = {
        "sessionStart": "session",
        "beforeSubmitPrompt": "prompt",
        "afterAgentThought": "thinking",
        "afterAgentResponse": "response",
        "preCompact": "compaction",
        "stop": "session.end",
        "sessionEnd": "session.end",
    }
    if hook in static:
        return static[hook]
    if hook == "postToolUse":
        return f"tool:{event.get('tool_name', 'unknown')}"
    if hook == "postToolUseFailure":
        return f"tool:{event.get('tool_name', 'unknown')}.error"
    if hook == "afterShellExecution":
        return "shell"
    if hook == "afterMCPExecution":
        return f"mcp:{event.get('tool_name', 'unknown')}"
    if hook == "afterFileEdit":
        fp = event.get("file_path", "unknown")
        return f"edit:{os.path.basename(fp)}"
    if hook == "subagentStop":
        return f"subagent:{event.get('subagent_type', 'unknown')}"
    return hook


def event_to_span_kind(event: dict) -> str:
    hook = event.get("hook_event_name", "")
    if hook in ("postToolUse", "postToolUseFailure", "afterShellExecution",
                "afterMCPExecution", "afterFileEdit"):
        return "TOOL"
    return "CHAIN"


def _json_str(v: object) -> str:
    return json.dumps(v) if isinstance(v, (dict, list)) else str(v)


def event_to_attributes(event: dict) -> dict:
    attrs: dict = {}
    hook = event.get("hook_event_name", "")

    cid = event.get("conversation_id")
    if cid:
        attrs["session.id"] = cid
    if "user_email" in event:
        attrs["user.id"] = str(event["user_email"])
    if "model" in event:
        attrs["llm.model_name"] = str(event["model"])

    for k in ("conversation_id", "generation_id", "hook_event_name", "cursor_version"):
        if k in event and event[k] is not None:
            attrs[k] = str(event[k])

    if hook == "sessionStart":
        for k in ("composer_mode", "is_background_agent"):
            if k in event:
                attrs[k] = str(event[k])

    elif hook == "beforeSubmitPrompt":
        if "prompt" in event:
            attrs["input.value"] = event["prompt"]
            attrs["input.mime_type"] = "text/plain"
        if "attachments" in event:
            attrs["attachments"] = json.dumps(event["attachments"])

    elif hook == "afterAgentThought":
        if "text" in event:
            attrs["output.value"] = event["text"]
            attrs["output.mime_type"] = "text/plain"
        if "duration_ms" in event:
            attrs["duration_ms"] = str(event["duration_ms"])

    elif hook == "afterAgentResponse":
        if "text" in event:
            attrs["output.value"] = event["text"]
            attrs["output.mime_type"] = "text/plain"

    elif hook == "postToolUse":
        if "tool_name" in event:
            attrs["tool.name"] = str(event["tool_name"])
        if "tool_input" in event:
            attrs["input.value"] = _json_str(event["tool_input"])
            attrs["input.mime_type"] = "application/json"
        if "tool_output" in event:
            attrs["output.value"] = _json_str(event["tool_output"])
            attrs["output.mime_type"] = "text/plain"
        if "duration" in event:
            attrs["duration"] = str(event["duration"])

    elif hook == "postToolUseFailure":
        if "tool_name" in event:
            attrs["tool.name"] = str(event["tool_name"])
        if "tool_input" in event:
            attrs["input.value"] = _json_str(event["tool_input"])
            attrs["input.mime_type"] = "application/json"
        for k in ("error_message", "failure_type", "duration"):
            if k in event:
                attrs[k] = _json_str(event[k])

    elif hook == "afterShellExecution":
        if "command" in event:
            attrs["input.value"] = str(event["command"])
            attrs["input.mime_type"] = "text/plain"
        if "output" in event:
            attrs["output.value"] = str(event["output"])
            attrs["output.mime_type"] = "text/plain"
        if "duration" in event:
            attrs["duration"] = str(event["duration"])

    elif hook == "afterMCPExecution":
        if "tool_name" in event:
            attrs["tool.name"] = str(event["tool_name"])
        if "tool_input" in event:
            attrs["input.value"] = _json_str(event["tool_input"])
            attrs["input.mime_type"] = "application/json"
        if "result_json" in event:
            attrs["output.value"] = str(event["result_json"])
            attrs["output.mime_type"] = "application/json"
        if "duration" in event:
            attrs["duration"] = str(event["duration"])

    elif hook == "afterFileEdit":
        if "file_path" in event:
            attrs["file_path"] = event["file_path"]
            attrs["input.value"] = event["file_path"]
            attrs["input.mime_type"] = "text/plain"
        if "edits" in event:
            attrs["output.value"] = json.dumps(event["edits"])
            attrs["output.mime_type"] = "application/json"

    elif hook == "preCompact":
        for k in ("context_tokens", "context_window_size", "context_usage_percent",
                   "message_count", "trigger"):
            if k in event:
                attrs[k] = str(event[k])

    elif hook == "subagentStop":
        if "task" in event:
            attrs["input.value"] = str(event["task"])
            attrs["input.mime_type"] = "text/plain"
        if "summary" in event:
            attrs["output.value"] = str(event["summary"])
            attrs["output.mime_type"] = "text/plain"
        for k in ("subagent_type", "status", "duration_ms",
                   "tool_call_count", "message_count"):
            if k in event:
                attrs[k] = str(event[k])

    elif hook in ("stop", "sessionEnd"):
        for k in ("status", "reason", "duration_ms"):
            if k in event:
                attrs[k] = str(event[k])

    return attrs


def find_session_labels(events: list[dict]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for e in events:
        cid = e.get("conversation_id")
        if cid and cid not in labels and e.get("hook_event_name") == "beforeSubmitPrompt":
            prompt = e.get("prompt", "")
            labels[cid] = prompt[:120] if prompt else "untitled"
    return labels


def assign_turns(events: list[dict]) -> None:
    """Split events into turns. Each beforeSubmitPrompt starts a new turn.
    Events before the first prompt (like sessionStart) go into turn 0.
    Each event gets _turn_index, _span_id, _parent_span_id, and _trace_id."""
    turn_counters: dict[str, int] = {}
    turn_root_spans: dict[str, str] = {}

    for e in events:
        cid = e.get("conversation_id")
        if not cid:
            e["_span_id"] = make_span_id()
            e["_turn_index"] = 0
            continue

        hook = e.get("hook_event_name", "")

        if hook == "beforeSubmitPrompt":
            turn_counters[cid] = turn_counters.get(cid, -1) + 1
            turn_idx = turn_counters[cid]
            root_sid = make_span_id()
            turn_root_spans[f"{cid}:{turn_idx}"] = root_sid
            e["_span_id"] = root_sid
        else:
            turn_idx = turn_counters.get(cid, 0)
            e["_span_id"] = make_span_id()
            root_key = f"{cid}:{turn_idx}"
            if root_key in turn_root_spans:
                e["_parent_span_id"] = turn_root_spans[root_key]

        e["_turn_index"] = turn_idx
        e["_trace_id"] = make_trace_id(cid, turn_idx)


def build_span(event: dict, session_label: str | None = None) -> dict:
    ts = event.get("_timestamp")
    if ts is None:
        ts = time.time()
    else:
        ts = float(ts)

    start_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    duration_ms = event.get("duration_ms") or event.get("duration") or 0
    try:
        duration_ms = float(duration_ms)
    except (ValueError, TypeError):
        duration_ms = 0
    end_ts = ts + (duration_ms / 1000.0) if duration_ms else ts + 0.001
    end_time = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()

    hook = event.get("hook_event_name", "")
    is_error = hook == "postToolUseFailure" or event.get("status") == "error"

    attrs = event_to_attributes(event)
    if session_label:
        attrs["session_label"] = session_label

    name = event_to_span_name(event)
    hook = event.get("hook_event_name", "")
    if hook == "beforeSubmitPrompt" and attrs.get("input.value"):
        name = attrs["input.value"][:120]

    trace_id = event.get("_trace_id") or make_trace_id(
        event.get("conversation_id", "unknown"),
        event.get("_turn_index", 0),
    )

    span: dict = {
        "name": name,
        "context": {
            "trace_id": trace_id,
            "span_id": event.get("_span_id", make_span_id()),
        },
        "span_kind": event_to_span_kind(event),
        "start_time": start_time,
        "end_time": end_time,
        "status_code": "ERROR" if is_error else "OK",
        "status_message": event.get("error_message", ""),
        "attributes": attrs,
    }

    parent_id = event.get("_parent_span_id")
    if parent_id:
        span["parent_id"] = parent_id

    return span


def post_to_phoenix(spans: list[dict]) -> bool:
    try:
        from phoenix.client import Client
        client = Client(base_url=PHOENIX_HOST)
        result = client.spans.log_spans(
            project_identifier=PHOENIX_PROJECT,
            spans=spans,
        )
        log(f"Phoenix SDK response: {result}")
        return True
    except Exception as e:
        log(f"Phoenix SDK error: {e}")
        return False


def main() -> None:
    if not os.path.exists(BUFFER_PATH):
        log("No buffer file found")
        return

    try:
        with open(BUFFER_PATH) as f:
            lines = f.readlines()
    except Exception as e:
        log(f"Error reading buffer: {e}")
        return

    if not lines:
        log("Buffer is empty")
        return

    events: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            log(f"Skipping malformed line: {e}")

    if not events:
        try:
            open(BUFFER_PATH, "w").close()
        except Exception:
            pass
        return

    log(f"Flushing {len(events)} events to Phoenix at {PHOENIX_HOST}")

    events = [redact(e) for e in events]
    session_labels = find_session_labels(events)
    assign_turns(events)

    last_response_per_turn: dict[str, str] = {}
    for e in events:
        cid = e.get("conversation_id", "")
        turn = e.get("_turn_index", 0)
        key = f"{cid}:{turn}"
        if e.get("hook_event_name") == "afterAgentResponse":
            text = e.get("text", "")
            if text:
                last_response_per_turn[key] = text[:200]

    spans = []
    for e in events:
        cid = e.get("conversation_id", "")
        label = session_labels.get(cid)
        span = build_span(e, session_label=label)
        hook = e.get("hook_event_name", "")
        turn = e.get("_turn_index", 0)
        turn_key = f"{cid}:{turn}"
        if hook == "beforeSubmitPrompt" and turn_key in last_response_per_turn:
            span["attributes"]["output.value"] = last_response_per_turn[turn_key]
            span["attributes"]["output.mime_type"] = "text/plain"
        spans.append(span)

    if post_to_phoenix(spans):
        try:
            open(BUFFER_PATH, "w").close()
            log("Buffer truncated after successful flush")
        except Exception as e:
            log(f"Error truncating buffer: {e}")
    else:
        log("Flush failed — buffer preserved for retry")


if __name__ == "__main__":
    main()
