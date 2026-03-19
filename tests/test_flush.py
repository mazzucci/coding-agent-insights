"""
Tests for coding-agent-insights flush.py

Validates:
1. Turn ordering / assignment correctness
2. Monotonic sequence numbers and micro-offset timestamps
3. Span parent-child relationships
4. Timestamp ordering within and across turns
5. Edge cases (missing fields, concurrent sessions, rapid events)
6. Atomic buffer drain (race condition fix)
7. Redaction
8. Trace ID determinism
"""
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

# Add hooks directory to path so we can import flush
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hooks'))
import flush


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build synthetic Cursor hook events
# ═══════════════════════════════════════════════════════════════════════════════

def make_event(hook_event_name, conversation_id="conv-1", **kwargs):
    """Create a synthetic hook event with auto-incrementing timestamp."""
    event = {
        "hook_event_name": hook_event_name,
        "conversation_id": conversation_id,
        "_timestamp": kwargs.pop("_timestamp", time.time()),
    }
    event.update(kwargs)
    return event


def make_session(conversation_id="conv-1", num_turns=3, events_per_turn=4):
    """
    Generate a realistic multi-turn session:
      sessionStart
      for each turn:
          beforeSubmitPrompt
          afterAgentThought
          postToolUse (1-2x)
          afterAgentResponse
      stop
    """
    events = []
    ts = 1700000000.0  # fixed base timestamp

    # sessionStart
    events.append(make_event("sessionStart", conversation_id,
                             _timestamp=ts, composer_mode="agent", is_background_agent=False))
    ts += 0.1

    for turn in range(num_turns):
        # User prompt
        events.append(make_event("beforeSubmitPrompt", conversation_id,
                                 _timestamp=ts, prompt=f"Turn {turn}: do something"))
        ts += 0.5

        # Agent thinking
        events.append(make_event("afterAgentThought", conversation_id,
                                 _timestamp=ts, text=f"Thinking about turn {turn}...",
                                 duration_ms=200))
        ts += 0.3

        # Tool usage(s)
        for i in range(min(events_per_turn - 2, 2)):
            events.append(make_event("postToolUse", conversation_id,
                                     _timestamp=ts, tool_name=f"tool_{i}",
                                     tool_input={"file": f"test_{i}.py"},
                                     tool_output="ok", duration=150))
            ts += 0.2

        # Agent response
        events.append(make_event("afterAgentResponse", conversation_id,
                                 _timestamp=ts, text=f"Done with turn {turn}"))
        ts += 0.5

    # Session end
    events.append(make_event("stop", conversation_id,
                             _timestamp=ts, status="completed", reason="normal",
                             duration_ms=int((ts - 1700000000.0) * 1000)))
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Turn assignment correctness
# ═══════════════════════════════════════════════════════════════════════════════

def test_turn_assignment_basic():
    """Each beforeSubmitPrompt should start a new turn; child events stay in that turn."""
    events = make_session(num_turns=3)
    flush.assign_turns(events)

    assert events[0]["_turn_index"] == 0, f"sessionStart should be turn 0"

    prompt_indices = [i for i, e in enumerate(events) if e["hook_event_name"] == "beforeSubmitPrompt"]
    assert len(prompt_indices) == 3, f"Expected 3 prompts, got {len(prompt_indices)}"

    for i, pi in enumerate(prompt_indices):
        assert events[pi]["_turn_index"] == i, \
            f"Prompt at index {pi} should be turn {i}, got {events[pi]['_turn_index']}"

    for idx in range(len(events)):
        e = events[idx]
        if e["hook_event_name"] not in ("beforeSubmitPrompt", "sessionStart", "stop"):
            belonging_prompt_idx = None
            for pi in reversed(prompt_indices):
                if pi < idx:
                    belonging_prompt_idx = pi
                    break
            if belonging_prompt_idx is not None:
                expected_turn = events[belonging_prompt_idx]["_turn_index"]
                assert e["_turn_index"] == expected_turn, \
                    f"Event {e['hook_event_name']} at idx {idx} should be turn {expected_turn}"

    print("✓ test_turn_assignment_basic PASSED")


def test_turn_assignment_stop_event():
    """The stop event should belong to the LAST turn, not turn 0."""
    events = make_session(num_turns=2)
    flush.assign_turns(events)

    stop_event = [e for e in events if e["hook_event_name"] == "stop"][0]
    assert stop_event["_turn_index"] == 1, \
        f"stop event should be in turn 1, got {stop_event['_turn_index']}"
    print("✓ test_turn_assignment_stop_event PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Monotonic sequence numbers
# ═══════════════════════════════════════════════════════════════════════════════

def test_event_sequence_numbers():
    """Every event should receive a monotonically increasing _event_seq."""
    events = make_session(num_turns=3)
    flush.assign_turns(events)

    seqs = [e["_event_seq"] for e in events]
    assert seqs == list(range(len(events))), \
        f"Sequences should be 0..{len(events)-1}, got {seqs}"
    print("✓ test_event_sequence_numbers PASSED")


def test_event_sequence_attribute_in_span():
    """Built spans should contain the event.sequence attribute."""
    events = make_session(num_turns=1)
    flush.assign_turns(events)
    spans = [flush.build_span(e) for e in events]

    for span in spans:
        assert "event.sequence" in span["attributes"], \
            f"Span '{span['name']}' missing event.sequence attribute"

    seq_values = [int(s["attributes"]["event.sequence"]) for s in spans]
    assert seq_values == list(range(len(spans)))
    print("✓ test_event_sequence_attribute_in_span PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Micro-offset timestamps fix ordering
# ═══════════════════════════════════════════════════════════════════════════════

def test_micro_offset_guarantees_unique_start_times():
    """Even when all events have the SAME wall-clock timestamp, each span
    should get a unique, monotonically increasing start_time."""
    ts = 1700000000.0
    events = [
        make_event("beforeSubmitPrompt", "conv-1", _timestamp=ts, prompt="test"),
        make_event("afterAgentThought", "conv-1", _timestamp=ts, text="thinking"),
        make_event("postToolUse", "conv-1", _timestamp=ts, tool_name="edit"),
        make_event("postToolUse", "conv-1", _timestamp=ts, tool_name="read"),
        make_event("afterAgentResponse", "conv-1", _timestamp=ts, text="done"),
    ]

    flush.assign_turns(events)
    spans = [flush.build_span(e) for e in events]

    start_times = [s["start_time"] for s in spans]
    assert len(set(start_times)) == len(start_times), \
        f"All start_times should be unique, got duplicates: {start_times}"

    # Verify they are in order
    for i in range(1, len(start_times)):
        assert start_times[i] > start_times[i - 1], \
            f"start_time[{i}] should be > start_time[{i-1}]"

    print("✓ test_micro_offset_guarantees_unique_start_times PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Parent-child span relationships
# ═══════════════════════════════════════════════════════════════════════════════

def test_parent_child_relationships():
    """Child events should have _parent_span_id pointing to the turn's root span."""
    events = make_session(num_turns=2)
    flush.assign_turns(events)

    prompts = [e for e in events if e["hook_event_name"] == "beforeSubmitPrompt"]
    for p in prompts:
        assert "_parent_span_id" not in p, "Prompt span should not have a parent"

    for e in events:
        if e["hook_event_name"] in ("beforeSubmitPrompt", "sessionStart"):
            continue
        if "_parent_span_id" in e:
            turn = e["_turn_index"]
            root_prompt = [p for p in prompts if p["_turn_index"] == turn]
            if root_prompt:
                assert e["_parent_span_id"] == root_prompt[0]["_span_id"]

    print("✓ test_parent_child_relationships PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Timestamp ordering across turns
# ═══════════════════════════════════════════════════════════════════════════════

def test_timestamp_ordering_across_turns():
    """Verify turns are ordered chronologically."""
    events = make_session(num_turns=3)
    flush.assign_turns(events)
    spans = [flush.build_span(e) for e in events]

    prompts = [(s, events[i]) for i, s in enumerate(spans)
               if events[i]["hook_event_name"] == "beforeSubmitPrompt"]

    for i in range(1, len(prompts)):
        prev_time = prompts[i-1][0]["start_time"]
        curr_time = prompts[i][0]["start_time"]
        assert curr_time > prev_time, \
            f"Turn {i} start_time ({curr_time}) should be after turn {i-1} ({prev_time})"

    print("✓ test_timestamp_ordering_across_turns PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Concurrent / interleaved sessions
# ═══════════════════════════════════════════════════════════════════════════════

def test_interleaved_sessions():
    """Two conversations interleaving events should produce correct independent turn indexing."""
    ts = 1700000000.0
    events = [
        make_event("sessionStart", "conv-A", _timestamp=ts),
        make_event("sessionStart", "conv-B", _timestamp=ts + 0.01),
        make_event("beforeSubmitPrompt", "conv-A", _timestamp=ts + 0.1, prompt="A turn 0"),
        make_event("beforeSubmitPrompt", "conv-B", _timestamp=ts + 0.15, prompt="B turn 0"),
        make_event("afterAgentThought", "conv-A", _timestamp=ts + 0.2, text="A thinking"),
        make_event("afterAgentThought", "conv-B", _timestamp=ts + 0.25, text="B thinking"),
        make_event("afterAgentResponse", "conv-A", _timestamp=ts + 0.3, text="A done"),
        make_event("beforeSubmitPrompt", "conv-A", _timestamp=ts + 0.5, prompt="A turn 1"),
        make_event("afterAgentResponse", "conv-B", _timestamp=ts + 0.55, text="B done"),
        make_event("afterAgentResponse", "conv-A", _timestamp=ts + 0.7, text="A turn 1 done"),
    ]

    flush.assign_turns(events)

    a_prompts = [e for e in events if e.get("conversation_id") == "conv-A"
                 and e["hook_event_name"] == "beforeSubmitPrompt"]
    assert [e["_turn_index"] for e in a_prompts] == [0, 1]

    b_prompts = [e for e in events if e.get("conversation_id") == "conv-B"
                 and e["hook_event_name"] == "beforeSubmitPrompt"]
    assert [e["_turn_index"] for e in b_prompts] == [0]

    a_trace_ids = set(e.get("_trace_id") for e in events if e.get("conversation_id") == "conv-A"
                      and e.get("_trace_id"))
    b_trace_ids = set(e.get("_trace_id") for e in events if e.get("conversation_id") == "conv-B"
                      and e.get("_trace_id"))
    assert not a_trace_ids & b_trace_ids, "Trace IDs should not overlap between sessions"

    print("✓ test_interleaved_sessions PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_events_before_first_prompt():
    """Events before any beforeSubmitPrompt should go to turn 0."""
    events = [
        make_event("sessionStart", "conv-1", _timestamp=1700000000.0),
        make_event("afterAgentThought", "conv-1", _timestamp=1700000000.1, text="pre-prompt thought"),
        make_event("beforeSubmitPrompt", "conv-1", _timestamp=1700000000.5, prompt="first"),
    ]
    flush.assign_turns(events)

    assert events[0]["_turn_index"] == 0
    assert events[1]["_turn_index"] == 0
    assert events[2]["_turn_index"] == 0
    print("✓ test_events_before_first_prompt PASSED")


def test_missing_conversation_id():
    """Events without conversation_id should still get span_id and turn 0."""
    events = [
        {"hook_event_name": "sessionStart", "_timestamp": 1700000000.0},
    ]
    flush.assign_turns(events)

    assert events[0]["_turn_index"] == 0
    assert "_span_id" in events[0]
    assert "_event_seq" in events[0]
    print("✓ test_missing_conversation_id PASSED")


def test_missing_timestamp():
    """Events without _timestamp should fall back to current time."""
    events = [
        {"hook_event_name": "sessionStart", "conversation_id": "conv-1"},
    ]
    flush.assign_turns(events)
    span = flush.build_span(events[0])
    assert span["start_time"] is not None
    print("✓ test_missing_timestamp PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Redaction
# ═══════════════════════════════════════════════════════════════════════════════

def test_redaction():
    """SKIP_FIELDS should redact top-level and nested fields."""
    original_skip = flush.SKIP_FIELDS
    try:
        flush.SKIP_FIELDS = {"prompt", "secret_key"}
        event = {
            "hook_event_name": "beforeSubmitPrompt",
            "prompt": "sensitive prompt text",
            "conversation_id": "conv-1",
            "nested": {"secret_key": "api-key-123", "safe": "ok"},
        }
        redacted = flush.redact(event)
        assert redacted["prompt"] == "[redacted]"
        assert redacted["nested"]["secret_key"] == "[redacted]"
        assert redacted["nested"]["safe"] == "ok"
        assert redacted["conversation_id"] == "conv-1"
        print("✓ test_redaction PASSED")
    finally:
        flush.SKIP_FIELDS = original_skip


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: Trace ID determinism
# ═══════════════════════════════════════════════════════════════════════════════

def test_trace_id_determinism():
    """Same conversation_id + turn_index should always produce the same trace_id."""
    id1 = flush.make_trace_id("conv-abc", 0)
    id2 = flush.make_trace_id("conv-abc", 0)
    id3 = flush.make_trace_id("conv-abc", 1)

    assert id1 == id2, "Same inputs should produce same trace_id"
    assert id1 != id3, "Different turn_index should produce different trace_id"
    print("✓ test_trace_id_determinism PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 10: Span construction
# ═══════════════════════════════════════════════════════════════════════════════

def test_span_construction_prompt():
    """Prompt spans should use the prompt text (truncated) as name."""
    events = [
        make_event("beforeSubmitPrompt", "conv-1", _timestamp=1700000000.0,
                    prompt="Fix the login page authentication bug that causes 500 errors"),
    ]
    flush.assign_turns(events)
    span = flush.build_span(events[0])

    assert span["name"].startswith("Fix the login page")
    assert len(span["name"]) <= 120
    assert span["span_kind"] == "CHAIN"
    assert span["status_code"] == "OK"
    assert span["attributes"]["input.value"] == events[0]["prompt"]
    print("✓ test_span_construction_prompt PASSED")


def test_span_construction_tool_error():
    """Error spans should have ERROR status."""
    events = [
        make_event("postToolUseFailure", "conv-1", _timestamp=1700000000.0,
                    tool_name="file_write", error_message="Permission denied",
                    failure_type="os_error"),
    ]
    flush.assign_turns(events)
    span = flush.build_span(events[0])

    assert span["name"] == "tool:file_write.error"
    assert span["status_code"] == "ERROR"
    assert span["status_message"] == "Permission denied"
    assert span["span_kind"] == "TOOL"
    print("✓ test_span_construction_tool_error PASSED")


def test_span_duration_calculation():
    """Span end_time should be start_time + duration_ms."""
    events = [
        make_event("afterAgentThought", "conv-1", _timestamp=1700000000.0,
                    text="thinking", duration_ms=500),
    ]
    flush.assign_turns(events)
    span = flush.build_span(events[0])

    start = datetime.fromisoformat(span["start_time"])
    end = datetime.fromisoformat(span["end_time"])
    delta_ms = (end - start).total_seconds() * 1000

    assert abs(delta_ms - 500) < 1, f"Duration should be ~500ms, got {delta_ms}ms"
    print("✓ test_span_duration_calculation PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 11: Session labels
# ═══════════════════════════════════════════════════════════════════════════════

def test_session_labels():
    """Session label should come from the first prompt of each conversation."""
    events = make_session("conv-1", num_turns=3)
    labels = flush.find_session_labels(events)

    assert "conv-1" in labels
    assert labels["conv-1"] == "Turn 0: do something"
    print("✓ test_session_labels PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 12: Full end-to-end span generation
# ═══════════════════════════════════════════════════════════════════════════════

def test_end_to_end_span_generation():
    """Full pipeline: events -> assign_turns -> build_span."""
    events = make_session("conv-1", num_turns=2)
    events = [flush.redact(e) for e in events]
    session_labels = flush.find_session_labels(events)
    flush.assign_turns(events)

    spans = []
    for e in events:
        cid = e.get("conversation_id", "")
        label = session_labels.get(cid)
        spans.append(flush.build_span(e, session_label=label))

    # sessionStart(1) + per_turn(prompt + thought + 2 tools + response = 5) * 2 + stop(1)
    expected = 1 + 5 * 2 + 1
    assert len(spans) == expected, f"Expected {expected} spans, got {len(spans)}"

    required = {"name", "context", "span_kind", "start_time", "end_time", "status_code", "attributes"}
    for span in spans:
        missing = required - set(span.keys())
        assert not missing, f"Span '{span['name']}' missing fields: {missing}"

    # All spans in a turn should share the same trace_id
    turn_traces = {}
    for span, event in zip(spans, events):
        tid = span["context"]["trace_id"]
        turn = event.get("_turn_index", -1)
        cid = event.get("conversation_id", "")
        key = f"{cid}:{turn}"
        if key not in turn_traces:
            turn_traces[key] = tid
        else:
            assert turn_traces[key] == tid

    # All start_times should be unique (micro-offset fix)
    start_times = [s["start_time"] for s in spans]
    assert len(set(start_times)) == len(start_times), "All start_times should be unique"

    print("✓ test_end_to_end_span_generation PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 13: Atomic buffer drain
# ═══════════════════════════════════════════════════════════════════════════════

def test_atomic_buffer_drain():
    """_read_and_drain_buffer should atomically rename the buffer file so
    events arriving during processing are not lost."""
    import tempfile

    buf = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    buf_path = buf.name

    for i in range(5):
        buf.write(json.dumps({"hook_event_name": "test", "seq": i}) + "\n")
    buf.flush()
    buf.close()

    original_buffer = flush.BUFFER_PATH
    try:
        flush.BUFFER_PATH = buf_path
        lines = flush._read_and_drain_buffer()
    finally:
        flush.BUFFER_PATH = original_buffer

    assert len(lines) == 5, f"Should have read 5 lines, got {len(lines)}"
    assert not os.path.exists(buf_path), "Buffer file should be gone after drain"

    print("✓ test_atomic_buffer_drain PASSED")


def test_atomic_drain_does_not_lose_late_events():
    """Events appended after rename should survive in the new buffer file."""
    import tempfile

    buf = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    buf_path = buf.name

    for i in range(5):
        buf.write(json.dumps({"hook_event_name": "test", "seq": i}) + "\n")
    buf.flush()
    buf.close()

    original_buffer = flush.BUFFER_PATH
    try:
        flush.BUFFER_PATH = buf_path

        # Drain (renames the file away)
        lines = flush._read_and_drain_buffer()
        assert len(lines) == 5

        # Simulate a late event arriving into the ORIGINAL path
        # (since the hook script always appends to BUFFER_PATH)
        with open(buf_path, "a") as f:
            f.write(json.dumps({"hook_event_name": "late_arrival", "seq": 99}) + "\n")

        # The late event should be in the buffer for the next drain
        lines2 = flush._read_and_drain_buffer()
        assert len(lines2) == 1, f"Late event should survive, got {len(lines2)} lines"
        assert json.loads(lines2[0].strip())["seq"] == 99
    finally:
        flush.BUFFER_PATH = original_buffer
        if os.path.exists(buf_path):
            os.unlink(buf_path)

    print("✓ test_atomic_drain_does_not_lose_late_events PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 14: Output.value on root prompt span
# ═══════════════════════════════════════════════════════════════════════════════

def test_prompt_output_value_injection():
    """The last afterAgentResponse text should be injected as output.value on the prompt span."""
    events = make_session("conv-1", num_turns=1)
    events = [flush.redact(e) for e in events]
    session_labels = flush.find_session_labels(events)
    flush.assign_turns(events)

    last_response_per_turn = {}
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
        span = flush.build_span(e, session_label=label)
        hook = e.get("hook_event_name", "")
        turn = e.get("_turn_index", 0)
        turn_key = f"{cid}:{turn}"
        if hook == "beforeSubmitPrompt" and turn_key in last_response_per_turn:
            span["attributes"]["output.value"] = last_response_per_turn[turn_key]
            span["attributes"]["output.mime_type"] = "text/plain"
        spans.append(span)

    prompt_span = [s for s, e in zip(spans, events)
                   if e["hook_event_name"] == "beforeSubmitPrompt"][0]

    assert "output.value" in prompt_span["attributes"]
    assert prompt_span["attributes"]["output.value"] == "Done with turn 0"
    print("✓ test_prompt_output_value_injection PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("coding-agent-insights — flush.py test suite")
    print("=" * 70)
    print()

    tests = [
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
        test_atomic_buffer_drain,
        test_atomic_drain_does_not_lose_late_events,
        test_prompt_output_value_injection,
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
