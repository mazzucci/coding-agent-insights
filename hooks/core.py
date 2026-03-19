# /// script
# requires-python = ">=3.10"
# dependencies = ["arize-phoenix-client>=2.0.0"]
# ///
"""
coding-agent-insights — core engine

Agent-agnostic trace-building and Phoenix posting logic. Adapters
(Cursor, Claude Code, etc.) normalise raw events into the NormalizedEvent
format, then hand them here for span construction and export.
"""
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

PHOENIX_HOST = os.environ.get("PHOENIX_HOST", "http://localhost:6006")
PHOENIX_PROJECT = os.environ.get("PHOENIX_PROJECT", "coding-agent-insights")
SKIP_FIELDS = set(
    f.strip()
    for f in os.environ.get("TRACES_SKIP", os.environ.get("CURSOR_TRACES_SKIP", "")).split(",")
    if f.strip()
)
DEBUG = os.environ.get("TRACES_DEBUG", os.environ.get("CURSOR_TRACES_DEBUG", "")).lower() in (
    "1",
    "true",
    "yes",
)
LOG_PATH = os.environ.get("TRACES_LOG", "/tmp/coding-agent-insights.log")


def log(msg: str) -> None:
    if not DEBUG:
        return
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")
    except Exception:
        pass


# ── Normalised event dataclass ────────────────────────────────────────────────


@dataclass
class NormalizedEvent:
    """Agent-agnostic representation of a single trace event.

    Adapters populate these fields; the core engine uses them to build
    OpenInference-compliant spans for Phoenix.
    """

    # Required
    event_type: str  # e.g. "prompt", "tool_use", "thinking", "response", "session_start", "session_end"
    conversation_id: str = ""
    timestamp: float = 0.0  # epoch seconds

    # Optional content
    name: str = ""  # span name override
    input_value: str = ""
    input_mime_type: str = "text/plain"
    output_value: str = ""
    output_mime_type: str = "text/plain"
    duration_ms: float = 0.0

    # Error info
    is_error: bool = False
    error_message: str = ""

    # Agent metadata
    agent_type: str = ""  # "cursor", "claude_code", etc.
    model: str = ""
    user_id: str = ""
    session_label: str = ""

    # Extra attributes (agent-specific)
    attributes: dict = field(default_factory=dict)

    # Internal — set by assign_turns()
    _turn_index: int = 0
    _event_seq: int = 0
    _span_id: str = ""
    _parent_span_id: str = ""
    _trace_id: str = ""


# ── Redaction ─────────────────────────────────────────────────────────────────


def redact_dict(d: dict) -> dict:
    """Redact fields listed in SKIP_FIELDS from a dict (recursive)."""
    if not SKIP_FIELDS:
        return d
    cleaned = {}
    for k, v in d.items():
        if k in SKIP_FIELDS:
            cleaned[k] = "[redacted]"
        elif isinstance(v, dict):
            cleaned[k] = redact_dict(v)
        else:
            cleaned[k] = v
    return cleaned


def redact_event(event: NormalizedEvent) -> NormalizedEvent:
    """Redact sensitive fields from a NormalizedEvent."""
    if not SKIP_FIELDS:
        return event
    for field_name in ("input_value", "output_value", "error_message", "name"):
        if field_name in SKIP_FIELDS:
            setattr(event, field_name, "[redacted]")
    event.attributes = redact_dict(event.attributes)
    return event


# ── ID generation ─────────────────────────────────────────────────────────────


def make_trace_id(conversation_id: str, turn_index: int = 0) -> str:
    key = f"{conversation_id}:turn:{turn_index}"
    return uuid.uuid5(uuid.NAMESPACE_URL, key).hex


def make_span_id() -> str:
    return uuid.uuid4().hex[:16]


# ── Span kind / name mapping ─────────────────────────────────────────────────

EVENT_TYPE_TO_SPAN_KIND = {
    "tool_use": "TOOL",
    "tool_error": "TOOL",
    "shell": "TOOL",
    "mcp": "TOOL",
    "file_edit": "TOOL",
}

EVENT_TYPE_TO_DEFAULT_NAME = {
    "session_start": "session",
    "prompt": "prompt",
    "thinking": "thinking",
    "response": "response",
    "compaction": "compaction",
    "session_end": "session.end",
    "subagent": "subagent",
}


def event_to_span_name(event: NormalizedEvent) -> str:
    if event.name:
        return event.name
    return EVENT_TYPE_TO_DEFAULT_NAME.get(event.event_type, event.event_type)


def event_to_span_kind(event: NormalizedEvent) -> str:
    return EVENT_TYPE_TO_SPAN_KIND.get(event.event_type, "CHAIN")


# ── Turn assignment ───────────────────────────────────────────────────────────


def assign_turns(events: list[NormalizedEvent]) -> None:
    """Split events into turns. Each 'prompt' event starts a new turn.
    Events before the first prompt go into turn 0.
    Each event gets _turn_index, _span_id, _parent_span_id, _trace_id,
    and _event_seq (monotonic counter for stable ordering in Phoenix)."""
    turn_counters: dict[str, int] = {}
    turn_root_spans: dict[str, str] = {}
    global_seq = 0

    for e in events:
        e._event_seq = global_seq
        global_seq += 1

        cid = e.conversation_id
        if not cid:
            e._span_id = make_span_id()
            e._turn_index = 0
            continue

        if e.event_type == "prompt":
            turn_counters[cid] = turn_counters.get(cid, -1) + 1
            turn_idx = turn_counters[cid]
            root_sid = make_span_id()
            turn_root_spans[f"{cid}:{turn_idx}"] = root_sid
            e._span_id = root_sid
        else:
            turn_idx = turn_counters.get(cid, 0)
            e._span_id = make_span_id()
            root_key = f"{cid}:{turn_idx}"
            if root_key in turn_root_spans:
                e._parent_span_id = turn_root_spans[root_key]

        e._turn_index = turn_idx
        e._trace_id = make_trace_id(cid, turn_idx)


# ── Session labels ────────────────────────────────────────────────────────────


def find_session_labels(events: list[NormalizedEvent]) -> dict[str, str]:
    """Extract session labels from the first prompt of each conversation."""
    labels: dict[str, str] = {}
    for e in events:
        cid = e.conversation_id
        if cid and cid not in labels and e.event_type == "prompt":
            labels[cid] = e.input_value[:120] if e.input_value else "untitled"
    return labels


# ── Span building ─────────────────────────────────────────────────────────────


def _json_str(v: object) -> str:
    return json.dumps(v) if isinstance(v, (dict, list)) else str(v)


def build_span(event: NormalizedEvent, session_label: str | None = None) -> dict:
    ts = event.timestamp if event.timestamp else time.time()

    # Micro-offset for monotonic ordering in Phoenix
    ts += event._event_seq * 0.0001

    start_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    duration_ms = event.duration_ms
    try:
        duration_ms = float(duration_ms)
    except (ValueError, TypeError):
        duration_ms = 0
    end_ts = ts + (duration_ms / 1000.0) if duration_ms else ts + 0.001
    end_time = datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat()

    # Build attributes
    attrs: dict = {}

    if event.conversation_id:
        attrs["session.id"] = event.conversation_id
    if event.user_id:
        attrs["user.id"] = event.user_id
    if event.model:
        attrs["llm.model_name"] = event.model
    if event.agent_type:
        attrs["agent.type"] = event.agent_type

    if event.input_value:
        attrs["input.value"] = event.input_value
        attrs["input.mime_type"] = event.input_mime_type
    if event.output_value:
        attrs["output.value"] = event.output_value
        attrs["output.mime_type"] = event.output_mime_type

    attrs["event.sequence"] = str(event._event_seq)

    if session_label:
        attrs["session_label"] = session_label

    # Merge adapter-specific attributes
    for k, v in event.attributes.items():
        attrs[k] = _json_str(v) if isinstance(v, (dict, list)) else str(v)

    # Span name — use prompt text for prompt events
    name = event_to_span_name(event)
    if event.event_type == "prompt" and event.input_value:
        name = event.input_value[:120]

    trace_id = event._trace_id or make_trace_id(
        event.conversation_id or "unknown",
        event._turn_index,
    )

    span: dict = {
        "name": name,
        "context": {
            "trace_id": trace_id,
            "span_id": event._span_id or make_span_id(),
        },
        "span_kind": event_to_span_kind(event),
        "start_time": start_time,
        "end_time": end_time,
        "status_code": "ERROR" if event.is_error else "OK",
        "status_message": event.error_message,
        "attributes": attrs,
    }

    if event._parent_span_id:
        span["parent_id"] = event._parent_span_id

    return span


# ── Phoenix posting ───────────────────────────────────────────────────────────


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


# ── Pipeline ──────────────────────────────────────────────────────────────────


def inject_output_on_prompts(
    events: list[NormalizedEvent], spans: list[dict]
) -> None:
    """Copy the last response text into the prompt span's output.value
    so Phoenix shows input→output on the root turn span."""
    last_response_per_turn: dict[str, str] = {}
    for e in events:
        key = f"{e.conversation_id}:{e._turn_index}"
        if e.event_type == "response" and e.output_value:
            last_response_per_turn[key] = e.output_value[:200]

    for span, event in zip(spans, events):
        if event.event_type == "prompt":
            key = f"{event.conversation_id}:{event._turn_index}"
            if key in last_response_per_turn:
                span["attributes"]["output.value"] = last_response_per_turn[key]
                span["attributes"]["output.mime_type"] = "text/plain"


def process_and_send(events: list[NormalizedEvent]) -> bool:
    """Full pipeline: redact → labels → turns → spans → post."""
    if not events:
        log("No events to process")
        return True

    events = [redact_event(e) for e in events]
    session_labels = find_session_labels(events)
    assign_turns(events)

    spans = []
    for e in events:
        label = session_labels.get(e.conversation_id)
        spans.append(build_span(e, session_label=label))

    inject_output_on_prompts(events, spans)

    log(f"Sending {len(spans)} spans to Phoenix at {PHOENIX_HOST}")
    if post_to_phoenix(spans):
        log(f"Successfully sent {len(spans)} spans")
        return True
    else:
        log("Failed to send spans to Phoenix")
        return False
