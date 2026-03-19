"""
coding-agent-insights — Claude Code adapter

Parses Claude Code JSONL session transcripts and normalises them into
NormalizedEvent objects that match the same trace structure as Cursor.

Claude Code transcript format:
  ~/.claude/projects/{project-path}/{sessionId}.jsonl
  Each line is a JSON object with fields:
    type: "user" | "assistant" | "result" | "summary"
    uuid, parentUuid, sessionId, timestamp
    message: { role, content: [...] }

Assistant content blocks have types:
    "text" — plain text response
    "thinking" — model reasoning
    "tool_use" — tool invocation (id, name, input)
    "tool_result" — comes in a subsequent "user" message (linked by tool_use_id)

Sub-agent transcripts: agent-{shortId}.jsonl with isSidechain: true
"""
import glob
import json
import os
from datetime import datetime, timezone

from hooks.core import NormalizedEvent, log


# Claude Code stores transcripts here
CLAUDE_PROJECTS_DIR = os.environ.get(
    "CLAUDE_PROJECTS_DIR",
    os.path.expanduser("~/.claude/projects"),
)


class ClaudeCodeAdapter:
    """Parse Claude Code JSONL transcripts into NormalizedEvents."""

    agent_type = "claude_code"

    def read_events(self, transcript_path: str | None = None) -> list[NormalizedEvent]:
        """Read a Claude Code transcript and return NormalizedEvents.

        Args:
            transcript_path: Path to a specific .jsonl transcript file.
                           If None, reads from CLAUDE_TRANSCRIPT_PATH env var.
        """
        path = transcript_path or os.environ.get("CLAUDE_TRANSCRIPT_PATH", "")
        if not path:
            log("Claude Code: no transcript path provided")
            return []

        if not os.path.exists(path):
            log(f"Claude Code: transcript not found: {path}")
            return []

        return self._parse_transcript(path)

    def read_session(self, session_id: str, project_path: str = "") -> list[NormalizedEvent]:
        """Read all transcripts for a given session ID.

        Searches the Claude projects directory for matching session files,
        including sub-agent sidechains.
        """
        if not project_path:
            project_path = CLAUDE_PROJECTS_DIR

        events = []

        # Find matching session files
        pattern = os.path.join(project_path, "**", f"{session_id}.jsonl")
        for filepath in glob.glob(pattern, recursive=True):
            events.extend(self._parse_transcript(filepath))

        # Also look for sub-agent transcripts
        base_dir = os.path.dirname(pattern) if events else project_path
        for filepath in glob.glob(
            os.path.join(project_path, "**", f"agent-*.jsonl"), recursive=True
        ):
            try:
                with open(filepath) as f:
                    first_line = f.readline()
                    if first_line:
                        first = json.loads(first_line)
                        if first.get("sessionId") == session_id:
                            events.extend(
                                self._parse_transcript(filepath, is_subagent=True)
                            )
            except (json.JSONDecodeError, OSError):
                continue

        events.sort(key=lambda e: e.timestamp)
        return events

    # ── Transcript parsing ────────────────────────────────────────────────────

    def _parse_transcript(
        self, path: str, is_subagent: bool = False
    ) -> list[NormalizedEvent]:
        """Parse a single JSONL transcript file into NormalizedEvents."""
        events: list[NormalizedEvent] = []
        pending_tool_uses: dict[str, NormalizedEvent] = {}  # tool_use_id → event
        session_id = ""

        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError as e:
            log(f"Claude Code: failed to read {path}: {e}")
            return []

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"Claude Code: malformed line {line_num} in {path}: {e}")
                continue

            if not session_id:
                session_id = raw.get("sessionId", os.path.basename(path).replace(".jsonl", ""))

            msg_type = raw.get("type", "")
            timestamp = self._parse_timestamp(raw.get("timestamp", ""))

            if msg_type == "user":
                user_events = self._parse_user_message(
                    raw, session_id, timestamp, pending_tool_uses, is_subagent
                )
                events.extend(user_events)

            elif msg_type == "assistant":
                assistant_events = self._parse_assistant_message(
                    raw, session_id, timestamp, pending_tool_uses, is_subagent
                )
                events.extend(assistant_events)

            elif msg_type == "summary":
                # Summary messages — session end marker
                events.append(
                    NormalizedEvent(
                        event_type="session_end",
                        conversation_id=session_id,
                        timestamp=timestamp,
                        agent_type="claude_code",
                        output_value=self._extract_text(raw.get("summary", "")),
                        attributes={"is_subagent": str(is_subagent)},
                    )
                )

        # Generate a session_start event from the first real event
        if events:
            events.insert(
                0,
                NormalizedEvent(
                    event_type="session_start",
                    conversation_id=session_id,
                    timestamp=events[0].timestamp - 0.001,
                    agent_type="claude_code",
                    attributes={
                        "is_subagent": str(is_subagent),
                        "transcript_path": path,
                    },
                ),
            )

        return events

    def _parse_user_message(
        self,
        raw: dict,
        session_id: str,
        timestamp: float,
        pending_tool_uses: dict[str, NormalizedEvent],
        is_subagent: bool,
    ) -> list[NormalizedEvent]:
        """Parse a 'user' type message.

        User messages can be:
        1. Actual user prompts (text content)
        2. Tool results (content[].type == "tool_result")
        """
        events: list[NormalizedEvent] = []
        message = raw.get("message", {})
        content = message.get("content", [])

        if isinstance(content, str):
            # Simple text prompt
            events.append(
                NormalizedEvent(
                    event_type="prompt",
                    conversation_id=session_id,
                    timestamp=timestamp,
                    agent_type="claude_code",
                    input_value=content,
                    attributes={"is_subagent": str(is_subagent)},
                )
            )
            return events

        has_tool_result = False
        prompt_text = ""

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")

            if block_type == "tool_result":
                has_tool_result = True
                tool_use_id = block.get("tool_use_id", "")
                result_content = block.get("content", "")
                is_error = block.get("is_error", False)

                # Update the pending tool_use event with its result
                if tool_use_id in pending_tool_uses:
                    tool_event = pending_tool_uses.pop(tool_use_id)
                    tool_event.output_value = self._extract_text(result_content)
                    tool_event.is_error = bool(is_error)
                    if is_error:
                        tool_event.error_message = tool_event.output_value[:500]
                        tool_event.event_type = "tool_error"
                    # Duration: from tool_use timestamp to this result timestamp
                    if timestamp > tool_event.timestamp:
                        tool_event.duration_ms = (timestamp - tool_event.timestamp) * 1000

            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    prompt_text += text + "\n"

        # If this user message has actual prompt text (not just tool results)
        if prompt_text.strip() and not has_tool_result:
            events.append(
                NormalizedEvent(
                    event_type="prompt",
                    conversation_id=session_id,
                    timestamp=timestamp,
                    agent_type="claude_code",
                    input_value=prompt_text.strip(),
                    attributes={"is_subagent": str(is_subagent)},
                )
            )

        return events

    def _parse_assistant_message(
        self,
        raw: dict,
        session_id: str,
        timestamp: float,
        pending_tool_uses: dict[str, NormalizedEvent],
        is_subagent: bool,
    ) -> list[NormalizedEvent]:
        """Parse an 'assistant' type message.

        Assistant content blocks:
        - text: model response text
        - thinking: model reasoning
        - tool_use: tool invocation (result comes later in a user message)
        """
        events: list[NormalizedEvent] = []
        message = raw.get("message", {})
        content = message.get("content", [])
        model = message.get("model", raw.get("model", ""))

        text_parts = []
        block_offset = 0.0

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")
            block_ts = timestamp + block_offset
            block_offset += 0.001  # Ensure ordering within a message

            if block_type == "thinking":
                events.append(
                    NormalizedEvent(
                        event_type="thinking",
                        conversation_id=session_id,
                        timestamp=block_ts,
                        agent_type="claude_code",
                        model=str(model),
                        output_value=block.get("thinking", ""),
                        attributes={"is_subagent": str(is_subagent)},
                    )
                )

            elif block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_input = block.get("input", {})
                tool_id = block.get("id", "")

                tool_event = NormalizedEvent(
                    event_type="tool_use",
                    conversation_id=session_id,
                    timestamp=block_ts,
                    agent_type="claude_code",
                    model=str(model),
                    name=f"tool:{tool_name}",
                    input_value=json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input),
                    input_mime_type="application/json",
                    attributes={
                        "tool.name": tool_name,
                        "tool_use_id": tool_id,
                        "is_subagent": str(is_subagent),
                    },
                )
                events.append(tool_event)

                # Track this tool_use so we can fill in the result later
                if tool_id:
                    pending_tool_uses[tool_id] = tool_event

            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)

        # Combine all text blocks into a single response event
        if text_parts:
            events.append(
                NormalizedEvent(
                    event_type="response",
                    conversation_id=session_id,
                    timestamp=timestamp + block_offset,
                    agent_type="claude_code",
                    model=str(model),
                    output_value="\n".join(text_parts),
                    attributes={"is_subagent": str(is_subagent)},
                )
            )

        return events

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_timestamp(ts_value) -> float:
        """Parse a Claude Code timestamp (ISO 8601 string or epoch ms) to epoch seconds."""
        if not ts_value:
            return 0.0

        if isinstance(ts_value, (int, float)):
            # Claude Code sometimes uses millisecond epoch
            if ts_value > 1e12:
                return ts_value / 1000.0
            return float(ts_value)

        if isinstance(ts_value, str):
            try:
                # ISO 8601 format
                dt = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                return dt.timestamp()
            except ValueError:
                pass
            try:
                return float(ts_value)
            except ValueError:
                pass

        return 0.0

    @staticmethod
    def _extract_text(content) -> str:
        """Extract plain text from various content formats."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif "text" in block:
                        parts.append(block["text"])
            return "\n".join(parts)
        return str(content) if content else ""
