---
name: coding-agent-insights
description: Flush pending traces, check status, search past sessions, and manage golden datasets in Phoenix.
---

# coding-agent-insights

## Trigger

Use this skill when the user asks to:
- Flush or send pending agent traces to Phoenix
- Check tracing status (buffer size, Phoenix health)
- Search or retrieve past agent sessions from Phoenix
- Add traces to a golden dataset for future reference

## Prerequisites

- `uv` must be installed (the flush script runs via `uv run`)
- Phoenix must be reachable (check the URL in `~/.cursor/hooks/.coding-agent-insights.env`)

## Flush pending traces

Run this to send all buffered events to Phoenix immediately:

```bash
uv run ~/.cursor/hooks/flush.py
```

To enable debug logging before flushing:

```bash
CURSOR_TRACES_DEBUG=true uv run ~/.cursor/hooks/flush.py
```

Debug output goes to `/tmp/cursor-traces.log`.

## Check status

Buffer size (number of pending events):

```bash
wc -l /tmp/cursor-traces.jsonl 2>/dev/null || echo "Buffer is empty"
```

Phoenix health:

```bash
curl -sf "$(grep PHOENIX_HOST ~/.cursor/hooks/.coding-agent-insights.env | cut -d'"' -f2)" && echo "OK" || echo "Unreachable"
```

## Search past traces

Use a Python script with `arize-phoenix-client` to query Phoenix. The Phoenix host and project are stored in `~/.cursor/hooks/.coding-agent-insights.env`.

### List recent sessions

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["arize-phoenix-client>=2.0.0"]
# ///
from phoenix.client import Client

client = Client(base_url="http://localhost:6006")  # or read from .coding-agent-insights.env

spans_df = client.spans.get_spans(project_identifier="cursor")
sessions = spans_df[spans_df["attributes"].apply(
    lambda a: "session.id" in a if isinstance(a, dict) else False
)]
print(sessions[["name", "start_time"]].head(20))
```

Run with: `uv run search_sessions.py`

### Search spans by keyword

To find spans matching a keyword (e.g. a tool name, error, or prompt fragment), filter the returned DataFrame:

```python
keyword = "your search term"
matches = spans_df[spans_df["name"].str.contains(keyword, case=False, na=False)]
print(matches[["name", "start_time", "status_code"]].head(20))
```

### Fetch all spans for a specific session

```python
session_id = "the-conversation-uuid"
session_spans = spans_df[spans_df["attributes"].apply(
    lambda a: a.get("session.id") == session_id if isinstance(a, dict) else False
)]
for _, span in session_spans.iterrows():
    print(f"  {span['name']}  ({span['status_code']})")
```

## Golden datasets

Phoenix datasets let you save exemplary traces for future reference — proven patterns, good prompts, or successful tool chains.

### Add spans to a dataset

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["arize-phoenix-client>=2.0.0"]
# ///
from phoenix.client import Client

client = Client(base_url="http://localhost:6006")

dataset = client.datasets.create(name="proven-patterns", description="Curated agent workflows")

examples = [
    {"input": {"prompt": "the original prompt"}, "output": {"response": "the agent response"}},
]
client.datasets.add_examples(dataset_id=dataset.id, examples=examples)
print(f"Added {len(examples)} examples to dataset '{dataset.name}'")
```

### List existing datasets

```python
datasets = client.datasets.list()
for ds in datasets:
    print(f"  {ds.name} — {ds.description} ({ds.example_count} examples)")
```

## Configuration

All settings live in `~/.cursor/hooks/.coding-agent-insights.env`:

| Variable | Default | Purpose |
|---|---|---|
| `PHOENIX_HOST` | `http://localhost:6006` | Phoenix server URL |
| `PHOENIX_PROJECT` | `cursor` | Phoenix project name |
| `CURSOR_TRACES_DEBUG` | _(unset)_ | Set to `true` for debug logging |
| `CURSOR_TRACES_SKIP` | _(unset)_ | Comma-separated field names to redact |
| `CURSOR_TRACES_BUFFER` | `/tmp/cursor-traces.jsonl` | Buffer file path |
