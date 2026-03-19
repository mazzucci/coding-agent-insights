"""
coding-agent-insights — adapter registry

Each adapter normalises agent-specific events into NormalizedEvent objects
that the core engine can process uniformly.
"""
from hooks.adapters.cursor import CursorAdapter
from hooks.adapters.claude_code import ClaudeCodeAdapter

ADAPTERS = {
    "cursor": CursorAdapter,
    "claude_code": ClaudeCodeAdapter,
}


def get_adapter(agent_type: str):
    """Return the adapter class for a given agent type."""
    adapter_cls = ADAPTERS.get(agent_type)
    if adapter_cls is None:
        raise ValueError(
            f"Unknown agent type: {agent_type!r}. "
            f"Available: {', '.join(ADAPTERS)}"
        )
    return adapter_cls()
