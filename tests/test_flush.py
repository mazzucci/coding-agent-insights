"""
Tests for coding-agent-insights

Validates:
  Core engine:
    1. Turn ordering / assignment correctness
    2. Monotonic sequence numbers and micro-offset timestamps
    3. Span parent-child relationships
    4. Timestamp ordering within and across turns
    5. Edge cases (missing fields, concurrent sessions, rapid events)
    6. Redaction
    7. Trace ID determinism
    8. Session labels
    9. Full pipeline (process_and_send)

  Cursor adapter:
    10. Event normalisation (all hook types)
    11. Atomic buffer drain (race condition fix)

  Claude Code adapter:
    12. JSONL transcript parsing
    13. Tool use / tool result pairing
    14. Thinking blocks
    15. Multi-turn conversations
    16. Sub-agent transcripts
    17. Timestamp parsing (ISO 8601 + epoch)
    18. End-to-end: transcript → NormalizedEvents → spans

  Cross-adapter consistency:
    19. Both adapters produce consistent span structures
"""
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hooks.core import (
    NormalizedEvent,
    assign_turns,
    build_span,
    find_session_labels,
    inject_output_on_prompts,
    make_span_id,
    make_trace_id,
    process_and_send,
    redact_dict,
    redact_event,
    SKIP_FIELDS,
)
from hooks.adapters.cursor import CursorAdapter
from hooks.adapters.claude_code import ClaudeCodeAdapter


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build synthetic NormalizedEvents
# ═══════════════════════════════════════════════════════════════════════════════

def make_event(event_type, conversation_id="conv-1", **kwargs):
    """Create a synthetic NormalizedEvent."""
    return NormalizedEvent(
        event_type=event_type,
        conversation_id=conversation_id,
        timestamp=kwargs.pop("timestamp", time.time()),
        agent_type=kwargs.pop("agent_type", "cursor"),
        **kwargs,
    )


def make_session(conversation_id="conv-1", num_turns=3, agent_type="cursor"):
    """Generate a realistic multi-turn session as NormalizedEvents."""
    events = []
    ts = 1700000000.0

    events.append(make_event(
        "session_start", conversation_id, timestamp=ts, agent_type=agent_type,
        attributes={"composer_mode": "agent"},
    ))
    ts += 0.1

    for turn in range(num_turns):
        events.append(make_event(
            "prompt", conversation_id, timestamp=ts, agent_type=agent_type,
            input_value=f"Turn {turn}: do something",
        ))
        ts += 0.5

        events.append(make_event(
            "thinking", conversation_id, timestamp=ts, agent_type=agent_type,
            output_value=f"Thinking about turn {turn}...", duration_ms=200,
        ))
        ts += 0.3

        for i in range(2):
            events.append(make_event(
                "tool_use", conversation_id, timestamp=ts, agent_type=agent_type,
                name=f"tool:tool_{i}",
                input_value=json.dumps({"file": f"test_{i}.py"}),
                input_mime_type="application/json",
                output_value="ok", duration_ms=150,
                attributes={"tool.name": f"tool_{i}"},
            ))
            ts += 0.2

        events.append(make_event(
            "response", conversation_id, timestamp=ts, agent_type=agent_type,
            output_value=f"Done with turn {turn}",
        ))
        ts += 0.5

    events.append(make_event(
        "session_end", conversation_id, timestamp=ts, agent_type=agent_type,
        attributes={"status": "completed", "reason": "normal"},
    ))
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_turn_assignment_basic():
    """Each prompt should start a new turn; child events stay in that turn."""
    events = make_session(num_turns=3)
    assign_turns(events)

    assert events[0]._turn_index == 0, "session_start should be turn 0"

    prompt_indices = [i for i, e in enumerate(events) if e.event_type == "prompt"]
    assert len(prompt_indices) == 3

    for i, pi in enumerate(prompt_indices):
        assert events[pi]._turn_index == i

    for idx in range(len(events)):
        e = events[idx]
        if e.event_type not in ("prompt", "session_start", "session_end"):
            belonging_prompt_idx = None
            for pi in reversed(prompt_indices):
                if pi < idx:
                    belonging_prompt_idx = pi
                    break
            if belonging_prompt_idx is not None:
                expected_turn = events[belonging_prompt_idx]._turn_index
                assert e._turn_index == expected_turn

    print("✓ test_turn_assignment_basic PASSED")


def test_turn_assignment_stop_event():
    """The session_end event should belong to the LAST turn."""
    events = make_session(num_turns=2)
    assign_turns(events)

    end_event = [e for e in events if e.event_type == "session_end"][0]
    assert end_event._turn_index == 1
    print("✓ test_turn_assignment_stop_event PASSED")


def test_event_sequence_numbers():
    """Every event should receive a monotonically increasing _event_seq."""
    events = make_session(num_turns=3)
    assign_turns(events)

    seqs = [e._event_seq for e in events]
    assert seqs == list(range(len(events)))
    print("✓ test_event_sequence_numbers PASSED")


def test_event_sequence_attribute_in_span():
    """Built spans should contain the event.sequence attribute."""
    events = make_session(num_turns=1)
    assign_turns(events)
    spans = [build_span(e) for e in events]

    for span in spans:
        assert "event.sequence" in span["attributes"]

    seq_values = [int(s["attributes"]["event.sequence"]) for s in spans]
    assert seq_values == list(range(len(spans)))
    print("✓ test_event_sequence_attribute_in_span PASSED")


def test_micro_offset_guarantees_unique_start_times():
    """Even when all events have the SAME timestamp, each span gets a unique start_time."""
    ts = 1700000000.0
    events = [
        make_event("prompt", "conv-1", timestamp=ts, input_value="test"),
        make_event("thinking", "conv-1", timestamp=ts, output_value="thinking"),
        make_event("tool_use", "conv-1", timestamp=ts, name="tool:edit"),
        make_event("tool_use", "conv-1", timestamp=ts, name="tool:read"),
        make_event("response", "conv-1", timestamp=ts, output_value="done"),
    ]

    assign_turns(events)
    spans = [build_span(e) for e in events]

    start_times = [s["start_time"] for s in spans]
    assert len(set(start_times)) == len(start_times), "All start_times should be unique"

    for i in range(1, len(start_times)):
        assert start_times[i] > start_times[i - 1]

    print("✓ test_micro_offset_guarantees_unique_start_times PASSED")


def test_parent_child_relationships():
    """Child events should have _parent_span_id pointing to the turn's root span."""
    events = make_session(num_turns=2)
    assign_turns(events)

    prompts = [e for e in events if e.event_type == "prompt"]
    for p in prompts:
        assert p._parent_span_id == "", "Prompt span should not have a parent"

    for e in events:
        if e.event_type in ("prompt", "session_start"):
            continue
        if e._parent_span_id:
            turn = e._turn_index
            root_prompt = [p for p in prompts if p._turn_index == turn]
            if root_prompt:
                assert e._parent_span_id == root_prompt[0]._span_id

    print("✓ test_parent_child_relationships PASSED")


def test_timestamp_ordering_across_turns():
    """Verify turns are ordered chronologically."""
    events = make_session(num_turns=3)
    assign_turns(events)
    spans = [build_span(e) for e in events]

    prompts = [(s, events[i]) for i, s in enumerate(spans)
               if events[i].event_type == "prompt"]

    for i in range(1, len(prompts)):
        prev_time = prompts[i-1][0]["start_time"]
        curr_time = prompts[i][0]["start_time"]
        assert curr_time > prev_time

    print("✓ test_timestamp_ordering_across_turns PASSED")


def test_interleaved_sessions():
    """Two conversations interleaving events should produce correct independent turn indexing."""
    ts = 1700000000.0
    events = [
        make_event("session_start", "conv-A", timestamp=ts),
        make_event("session_start", "conv-B", timestamp=ts + 0.01),
        make_event("prompt", "conv-A", timestamp=ts + 0.1, input_value="A turn 0"),
        make_event("prompt", "conv-B", timestamp=ts + 0.15, input_value="B turn 0"),
        make_event("thinking", "conv-A", timestamp=ts + 0.2, output_value="A thinking"),
        make_event("thinking", "conv-B", timestamp=ts + 0.25, output_value="B thinking"),
        make_event("response", "conv-A", timestamp=ts + 0.3, output_value="A done"),
        make_event("prompt", "conv-A", timestamp=ts + 0.5, input_value="A turn 1"),
        make_event("response", "conv-B", timestamp=ts + 0.55, output_value="B done"),
        make_event("response", "conv-A", timestamp=ts + 0.7, output_value="A turn 1 done"),
    ]

    assign_turns(events)

    a_prompts = [e for e in events if e.conversation_id == "conv-A" and e.event_type == "prompt"]
    assert [e._turn_index for e in a_prompts] == [0, 1]

    b_prompts = [e for e in events if e.conversation_id == "conv-B" and e.event_type == "prompt"]
    assert [e._turn_index for e in b_prompts] == [0]

    a_trace_ids = set(e._trace_id for e in events if e.conversation_id == "conv-A" and e._trace_id)
    b_trace_ids = set(e._trace_id for e in events if e.conversation_id == "conv-B" and e._trace_id)
    assert not a_trace_ids & b_trace_ids

    print("✓ test_interleaved_sessions PASSED")


def test_events_before_first_prompt():
    """Events before any prompt should go to turn 0."""
    events = [
        make_event("session_start", "conv-1", timestamp=1700000000.0),
        make_event("thinking", "conv-1", timestamp=1700000000.1, output_value="pre-prompt"),
        make_event("prompt", "conv-1", timestamp=1700000000.5, input_value="first"),
    ]
    assign_turns(events)

    assert events[0]._turn_index == 0
    assert events[1]._turn_index == 0
    assert events[2]._turn_index == 0
    print("✓ test_events_before_first_prompt PASSED")


def test_missing_conversation_id():
    """Events without conversation_id should still get span_id and turn 0."""
    events = [NormalizedEvent(event_type="session_start", timestamp=1700000000.0)]
    assign_turns(events)

    assert events[0]._turn_index == 0
    assert events[0]._span_id != ""
    print("✓ test_missing_conversation_id PASSED")


def test_missing_timestamp():
    """Events without timestamp should fall back to current time."""
    events = [NormalizedEvent(event_type="session_start", conversation_id="conv-1")]
    assign_turns(events)
    span = build_span(events[0])
    assert span["start_time"] is not None
    print("✓ test_missing_timestamp PASSED")


def test_redaction():
    """SKIP_FIELDS should redact fields."""
    import hooks.core as core_mod
    original_skip = core_mod.SKIP_FIELDS
    try:
        core_mod.SKIP_FIELDS = {"input_value", "secret_key"}

        event = NormalizedEvent(
            event_type="prompt",
            conversation_id="conv-1",
            input_value="sensitive prompt text",
            attributes={"secret_key": "api-key-123", "safe": "ok"},
        )
        redacted = redact_event(event)
        assert redacted.input_value == "[redacted]"
        assert redacted.attributes["secret_key"] == "[redacted]"
        assert redacted.attributes["safe"] == "ok"
        print("✓ test_redaction PASSED")
    finally:
        core_mod.SKIP_FIELDS = original_skip


def test_trace_id_determinism():
    """Same conversation_id + turn_index should always produce the same trace_id."""
    id1 = make_trace_id("conv-abc", 0)
    id2 = make_trace_id("conv-abc", 0)
    id3 = make_trace_id("conv-abc", 1)

    assert id1 == id2
    assert id1 != id3
    print("✓ test_trace_id_determinism PASSED")


def test_span_construction_prompt():
    """Prompt spans should use the prompt text (truncated) as name."""
    events = [make_event(
        "prompt", "conv-1", timestamp=1700000000.0,
        input_value="Fix the login page authentication bug that causes 500 errors",
    )]
    assign_turns(events)
    span = build_span(events[0])

    assert span["name"].startswith("Fix the login page")
    assert len(span["name"]) <= 120
    assert span["span_kind"] == "CHAIN"
    assert span["status_code"] == "OK"
    assert span["attributes"]["input.value"] == events[0].input_value
    print("✓ test_span_construction_prompt PASSED")


def test_span_construction_tool_error():
    """Error spans should have ERROR status."""
    events = [make_event(
        "tool_error", "conv-1", timestamp=1700000000.0,
        name="tool:file_write.error",
        is_error=True, error_message="Permission denied",
        attributes={"tool.name": "file_write", "failure_type": "os_error"},
    )]
    assign_turns(events)
    span = build_span(events[0])

    assert span["name"] == "tool:file_write.error"
    assert span["status_code"] == "ERROR"
    assert span["status_message"] == "Permission denied"
    assert span["span_kind"] == "TOOL"
    print("✓ test_span_construction_tool_error PASSED")


def test_span_duration_calculation():
    """Span end_time should be start_time + duration_ms."""
    events = [make_event(
        "thinking", "conv-1", timestamp=1700000000.0,
        output_value="thinking", duration_ms=500,
    )]
    assign_turns(events)
    span = build_span(events[0])

    start = datetime.fromisoformat(span["start_time"])
    end = datetime.fromisoformat(span["end_time"])
    delta_ms = (end - start).total_seconds() * 1000

    assert abs(delta_ms - 500) < 1
    print("✓ test_span_duration_calculation PASSED")


def test_session_labels():
    """Session label should come from the first prompt of each conversation."""
    events = make_session("conv-1", num_turns=3)
    labels = find_session_labels(events)

    assert "conv-1" in labels
    assert labels["conv-1"] == "Turn 0: do something"
    print("✓ test_session_labels PASSED")


def test_end_to_end_span_generation():
    """Full pipeline: events -> assign_turns -> build_span."""
    events = make_session("conv-1", num_turns=2)
    session_labels = find_session_labels(events)
    assign_turns(events)

    spans = []
    for e in events:
        label = session_labels.get(e.conversation_id)
        spans.append(build_span(e, session_label=label))

    # session_start(1) + per_turn(prompt + thinking + 2 tools + response = 5) * 2 + session_end(1)
    expected = 1 + 5 * 2 + 1
    assert len(spans) == expected, f"Expected {expected} spans, got {len(spans)}"

    required = {"name", "context", "span_kind", "start_time", "end_time", "status_code", "attributes"}
    for span in spans:
        missing = required - set(span.keys())
        assert not missing

    start_times = [s["start_time"] for s in spans]
    assert len(set(start_times)) == len(start_times)

    print("✓ test_end_to_end_span_generation PASSED")


def test_prompt_output_value_injection():
    """The last response text should be injected as output.value on the prompt span."""
    events = make_session("conv-1", num_turns=1)
    session_labels = find_session_labels(events)
    assign_turns(events)

    spans = [build_span(e, session_label=session_labels.get(e.conversation_id)) for e in events]
    inject_output_on_prompts(events, spans)

    prompt_span = [s for s, e in zip(spans, events) if e.event_type == "prompt"][0]
    assert "output.value" in prompt_span["attributes"]
    assert prompt_span["attributes"]["output.value"] == "Done with turn 0"
    print("✓ test_prompt_output_value_injection PASSED")


def test_agent_type_in_span():
    """Spans should include agent.type attribute."""
    events = [make_event("prompt", "conv-1", timestamp=1700000000.0,
                         input_value="test", agent_type="cursor")]
    assign_turns(events)
    span = build_span(events[0])
    assert span["attributes"]["agent.type"] == "cursor"
    print("✓ test_agent_type_in_span PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Cursor Adapter Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_cursor_normalise_prompt():
    """Cursor adapter should normalise beforeSubmitPrompt events."""
    adapter = CursorAdapter()
    raw = {
        "hook_event_name": "beforeSubmitPrompt",
        "conversation_id": "conv-1",
        "_timestamp": 1700000000.0,
        "prompt": "fix the bug",
        "user_email": "user@example.com",
        "model": "gpt-4",
    }
    event = adapter._normalise(raw)

    assert event.event_type == "prompt"
    assert event.conversation_id == "conv-1"
    assert event.input_value == "fix the bug"
    assert event.agent_type == "cursor"
    assert event.user_id == "user@example.com"
    assert event.model == "gpt-4"
    print("✓ test_cursor_normalise_prompt PASSED")


def test_cursor_normalise_tool_use():
    """Cursor adapter should normalise postToolUse events."""
    adapter = CursorAdapter()
    raw = {
        "hook_event_name": "postToolUse",
        "conversation_id": "conv-1",
        "_timestamp": 1700000000.0,
        "tool_name": "file_read",
        "tool_input": {"path": "test.py"},
        "tool_output": "content here",
        "duration": 150,
    }
    event = adapter._normalise(raw)

    assert event.event_type == "tool_use"
    assert event.name == "tool:file_read"
    assert "test.py" in event.input_value
    assert event.output_value == "content here"
    assert event.attributes["tool.name"] == "file_read"
    print("✓ test_cursor_normalise_tool_use PASSED")


def test_cursor_normalise_tool_failure():
    """Cursor adapter should normalise postToolUseFailure events with error status."""
    adapter = CursorAdapter()
    raw = {
        "hook_event_name": "postToolUseFailure",
        "conversation_id": "conv-1",
        "_timestamp": 1700000000.0,
        "tool_name": "file_write",
        "error_message": "Permission denied",
        "failure_type": "os_error",
    }
    event = adapter._normalise(raw)

    assert event.event_type == "tool_error"
    assert event.is_error is True
    assert event.error_message == "Permission denied"
    assert event.name == "tool:file_write.error"
    print("✓ test_cursor_normalise_tool_failure PASSED")


def test_cursor_normalise_all_hook_types():
    """Cursor adapter should handle all known hook event types."""
    adapter = CursorAdapter()
    hook_types = {
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
    for hook, expected_type in hook_types.items():
        raw = {
            "hook_event_name": hook,
            "conversation_id": "conv-1",
            "_timestamp": 1700000000.0,
        }
        event = adapter._normalise(raw)
        assert event.event_type == expected_type, f"{hook} → {event.event_type}, expected {expected_type}"

    print("✓ test_cursor_normalise_all_hook_types PASSED")


def test_cursor_atomic_buffer_drain():
    """Cursor adapter should atomically rename the buffer file."""
    adapter = CursorAdapter()

    buf = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    buf_path = buf.name
    for i in range(5):
        buf.write(json.dumps({"hook_event_name": "test", "conversation_id": "c1", "seq": i}) + "\n")
    buf.flush()
    buf.close()

    import hooks.adapters.cursor as cursor_mod
    original_buffer = cursor_mod.BUFFER_PATH
    try:
        cursor_mod.BUFFER_PATH = buf_path
        lines = adapter._read_and_drain_buffer()
    finally:
        cursor_mod.BUFFER_PATH = original_buffer

    assert len(lines) == 5
    assert not os.path.exists(buf_path)
    print("✓ test_cursor_atomic_buffer_drain PASSED")


def test_cursor_atomic_drain_late_events():
    """Events appended after rename should survive for the next drain."""
    adapter = CursorAdapter()

    buf = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    buf_path = buf.name
    for i in range(5):
        buf.write(json.dumps({"hook_event_name": "test", "conversation_id": "c1", "seq": i}) + "\n")
    buf.flush()
    buf.close()

    import hooks.adapters.cursor as cursor_mod
    original_buffer = cursor_mod.BUFFER_PATH
    try:
        cursor_mod.BUFFER_PATH = buf_path
        lines = adapter._read_and_drain_buffer()
        assert len(lines) == 5

        with open(buf_path, "a") as f:
            f.write(json.dumps({"hook_event_name": "late_arrival", "conversation_id": "c1", "seq": 99}) + "\n")

        lines2 = adapter._read_and_drain_buffer()
        assert len(lines2) == 1
        assert json.loads(lines2[0].strip())["seq"] == 99
    finally:
        cursor_mod.BUFFER_PATH = original_buffer
        if os.path.exists(buf_path):
            os.unlink(buf_path)

    print("✓ test_cursor_atomic_drain_late_events PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Claude Code Adapter Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _write_transcript(lines: list[dict]) -> str:
    """Write a list of dicts as JSONL to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    for line in lines:
        f.write(json.dumps(line) + "\n")
    f.flush()
    f.close()
    return f.name


def test_claude_code_simple_conversation():
    """Claude Code adapter should parse a simple user→assistant conversation."""
    transcript = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:00Z",
            "message": {"role": "user", "content": "Fix the login bug"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "text", "text": "I'll fix the login bug for you."},
                ],
            },
        },
    ]
    path = _write_transcript(transcript)
    try:
        adapter = ClaudeCodeAdapter()
        events = adapter.read_events(transcript_path=path)

        # Should have: session_start, prompt, response
        assert len(events) == 3, f"Expected 3 events, got {len(events)}"
        assert events[0].event_type == "session_start"
        assert events[1].event_type == "prompt"
        assert events[1].input_value == "Fix the login bug"
        assert events[2].event_type == "response"
        assert "fix the login bug" in events[2].output_value.lower()
        assert events[2].model == "claude-sonnet-4-20250514"
        assert all(e.agent_type == "claude_code" for e in events)
    finally:
        os.unlink(path)

    print("✓ test_claude_code_simple_conversation PASSED")


def test_claude_code_tool_use_and_result():
    """Claude Code adapter should pair tool_use with tool_result."""
    transcript = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:00Z",
            "message": {"role": "user", "content": "Read test.py"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:02Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-123",
                        "name": "Read",
                        "input": {"file_path": "test.py"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:04Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-123",
                        "content": "def test_hello():\n    pass",
                    },
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:06Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "text", "text": "I see the test file."},
                ],
            },
        },
    ]
    path = _write_transcript(transcript)
    try:
        adapter = ClaudeCodeAdapter()
        events = adapter.read_events(transcript_path=path)

        tool_events = [e for e in events if e.event_type == "tool_use"]
        assert len(tool_events) == 1
        assert tool_events[0].name == "tool:Read"
        assert tool_events[0].attributes["tool.name"] == "Read"
        assert "test.py" in tool_events[0].input_value
        # Tool result should be paired
        assert "def test_hello" in tool_events[0].output_value
        # Duration should be computed from tool_use to tool_result timestamp
        assert tool_events[0].duration_ms > 0
    finally:
        os.unlink(path)

    print("✓ test_claude_code_tool_use_and_result PASSED")


def test_claude_code_tool_error():
    """Claude Code adapter should handle tool errors."""
    transcript = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:00Z",
            "message": {"role": "user", "content": "Write to /etc/test"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:02Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-456",
                        "name": "Write",
                        "input": {"file_path": "/etc/test", "content": "hello"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:03Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-456",
                        "content": "Permission denied",
                        "is_error": True,
                    },
                ],
            },
        },
    ]
    path = _write_transcript(transcript)
    try:
        adapter = ClaudeCodeAdapter()
        events = adapter.read_events(transcript_path=path)

        tool_events = [e for e in events if e.event_type in ("tool_use", "tool_error")]
        assert len(tool_events) == 1
        assert tool_events[0].is_error is True
        assert tool_events[0].event_type == "tool_error"
        assert "Permission denied" in tool_events[0].error_message
    finally:
        os.unlink(path)

    print("✓ test_claude_code_tool_error PASSED")


def test_claude_code_thinking_blocks():
    """Claude Code adapter should extract thinking blocks."""
    transcript = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:00Z",
            "message": {"role": "user", "content": "Explain recursion"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:03Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "thinking", "thinking": "I should explain recursion with an example..."},
                    {"type": "text", "text": "Recursion is when a function calls itself."},
                ],
            },
        },
    ]
    path = _write_transcript(transcript)
    try:
        adapter = ClaudeCodeAdapter()
        events = adapter.read_events(transcript_path=path)

        thinking = [e for e in events if e.event_type == "thinking"]
        assert len(thinking) == 1
        assert "example" in thinking[0].output_value

        responses = [e for e in events if e.event_type == "response"]
        assert len(responses) == 1
        assert "Recursion" in responses[0].output_value
    finally:
        os.unlink(path)

    print("✓ test_claude_code_thinking_blocks PASSED")


def test_claude_code_multi_turn():
    """Claude Code adapter should handle multi-turn conversations."""
    transcript = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:00Z",
            "message": {"role": "user", "content": "Turn 1 prompt"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "Turn 1 response"}],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:01:00Z",
            "message": {"role": "user", "content": "Turn 2 prompt"},
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:01:05Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "Turn 2 response"}],
            },
        },
    ]
    path = _write_transcript(transcript)
    try:
        adapter = ClaudeCodeAdapter()
        events = adapter.read_events(transcript_path=path)

        prompts = [e for e in events if e.event_type == "prompt"]
        assert len(prompts) == 2
        assert prompts[0].input_value == "Turn 1 prompt"
        assert prompts[1].input_value == "Turn 2 prompt"

        # After assign_turns, they should be in different turns
        assign_turns(events)
        assert prompts[0]._turn_index == 0
        assert prompts[1]._turn_index == 1
    finally:
        os.unlink(path)

    print("✓ test_claude_code_multi_turn PASSED")


def test_claude_code_timestamp_parsing():
    """Claude Code adapter should parse various timestamp formats."""
    adapter = ClaudeCodeAdapter()

    # ISO 8601
    ts1 = adapter._parse_timestamp("2025-01-15T10:00:00Z")
    assert ts1 > 0

    # ISO 8601 with timezone offset
    ts2 = adapter._parse_timestamp("2025-01-15T10:00:00+00:00")
    assert abs(ts1 - ts2) < 1

    # Epoch milliseconds
    ts3 = adapter._parse_timestamp(1705312800000)
    assert ts3 > 1e9  # Should be in seconds, not milliseconds

    # Epoch seconds
    ts4 = adapter._parse_timestamp(1705312800)
    assert abs(ts3 - ts4) < 1

    # Empty / None
    assert adapter._parse_timestamp("") == 0.0
    assert adapter._parse_timestamp(None) == 0.0

    print("✓ test_claude_code_timestamp_parsing PASSED")


def test_claude_code_empty_transcript():
    """Claude Code adapter should handle empty/missing transcripts gracefully."""
    adapter = ClaudeCodeAdapter()

    # Non-existent file
    events = adapter.read_events(transcript_path="/nonexistent/path.jsonl")
    assert events == []

    # Empty file
    path = _write_transcript([])
    try:
        events = adapter.read_events(transcript_path=path)
        assert events == []
    finally:
        os.unlink(path)

    print("✓ test_claude_code_empty_transcript PASSED")


def test_claude_code_end_to_end_spans():
    """Full pipeline: Claude Code transcript → NormalizedEvents → spans."""
    transcript = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:00Z",
            "message": {"role": "user", "content": "Fix the bug in auth.py"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:02Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "thinking", "thinking": "Let me look at auth.py first"},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "auth.py"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:04Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "def login():\n    return None  # BUG",
                    },
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "sessionId": "sess-1",
            "timestamp": "2025-01-15T10:00:06Z",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [
                    {"type": "text", "text": "I found and fixed the bug."},
                ],
            },
        },
    ]
    path = _write_transcript(transcript)
    try:
        adapter = ClaudeCodeAdapter()
        events = adapter.read_events(transcript_path=path)
        assert len(events) > 0

        # Verify all events have agent_type
        for e in events:
            assert e.agent_type == "claude_code"

        # Run through full pipeline
        session_labels = find_session_labels(events)
        assign_turns(events)
        spans = [build_span(e, session_label=session_labels.get(e.conversation_id)) for e in events]
        inject_output_on_prompts(events, spans)

        # All spans should have required fields
        required = {"name", "context", "span_kind", "start_time", "end_time", "status_code", "attributes"}
        for span in spans:
            missing = required - set(span.keys())
            assert not missing, f"Span '{span['name']}' missing: {missing}"

        # All spans in the same turn should share a trace_id
        trace_ids = set(s["context"]["trace_id"] for s in spans)
        # One turn = one trace_id (plus possibly session_start in turn 0)
        assert len(trace_ids) <= 2

        # Start times should be unique
        start_times = [s["start_time"] for s in spans]
        assert len(set(start_times)) == len(start_times)

        # The prompt span should have agent.type = claude_code
        prompt_spans = [s for s in spans if "Fix the bug" in s["name"]]
        assert len(prompt_spans) == 1
        assert prompt_spans[0]["attributes"]["agent.type"] == "claude_code"
    finally:
        os.unlink(path)

    print("✓ test_claude_code_end_to_end_spans PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-adapter consistency tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_cursor_and_claude_code_span_parity():
    """Both adapters should produce spans with the same structure."""
    # Cursor-style events
    cursor_events = make_session("conv-cursor", num_turns=1, agent_type="cursor")
    assign_turns(cursor_events)
    cursor_spans = [build_span(e) for e in cursor_events]

    # Claude Code-style events (manually constructed to match)
    cc_events = make_session("conv-claude", num_turns=1, agent_type="claude_code")
    assign_turns(cc_events)
    cc_spans = [build_span(e) for e in cc_events]

    # Both should produce spans with identical required fields
    required = {"name", "context", "span_kind", "start_time", "end_time", "status_code", "attributes"}
    for spans_set, label in [(cursor_spans, "cursor"), (cc_spans, "claude_code")]:
        for span in spans_set:
            missing = required - set(span.keys())
            assert not missing, f"{label} span '{span['name']}' missing: {missing}"

    # Context should have trace_id and span_id
    for spans_set in [cursor_spans, cc_spans]:
        for span in spans_set:
            assert "trace_id" in span["context"]
            assert "span_id" in span["context"]

    # Agent type should be preserved
    for span in cursor_spans:
        assert span["attributes"].get("agent.type") == "cursor"
    for span in cc_spans:
        assert span["attributes"].get("agent.type") == "claude_code"

    print("✓ test_cursor_and_claude_code_span_parity PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("coding-agent-insights — test suite")
    print("=" * 70)
    print()

    tests = [
        # Core engine
        test_turn_assignment_basic,
        test_turn_assignment_stop_event,
        test_event_sequence_numbers,
        test_event_sequence_attribute_in_span,
        test_micro_offset_guarantees_unique_start_times,
        test_parent_child_relationships,
        test_timestamp_ordering_across_turns,
        test_interleaved_sessions,
        test_events_before_first_prompt,
        test_missing_conversation_id,
        test_missing_timestamp,
        test_redaction,
        test_trace_id_determinism,
        test_span_construction_prompt,
        test_span_construction_tool_error,
        test_span_duration_calculation,
        test_session_labels,
        test_end_to_end_span_generation,
        test_prompt_output_value_injection,
        test_agent_type_in_span,
        # Cursor adapter
        test_cursor_normalise_prompt,
        test_cursor_normalise_tool_use,
        test_cursor_normalise_tool_failure,
        test_cursor_normalise_all_hook_types,
        test_cursor_atomic_buffer_drain,
        test_cursor_atomic_drain_late_events,
        # Claude Code adapter
        test_claude_code_simple_conversation,
        test_claude_code_tool_use_and_result,
        test_claude_code_tool_error,
        test_claude_code_thinking_blocks,
        test_claude_code_multi_turn,
        test_claude_code_timestamp_parsing,
        test_claude_code_empty_transcript,
        test_claude_code_end_to_end_spans,
        # Cross-adapter
        test_cursor_and_claude_code_span_parity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n--- {test.__name__} ---")
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 70}")
    exit(0 if failed == 0 else 1)
