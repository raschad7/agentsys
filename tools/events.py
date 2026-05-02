"""In-process event bus for streaming live agent activity to the dashboard.

Each pipeline run gets a unique ``run_id`` and a dedicated FIFO queue.
Agents push small JSON-serialisable events; the SSE endpoint drains the
queue and forwards them to the browser.

The bus is intentionally tiny (no broker, no persistence) — it only needs
to live as long as a single dashboard request.
"""

from __future__ import annotations

import queue
import time
import uuid
from typing import Any, Optional

# run_id -> Queue[event dict]
_runs: dict[str, "queue.Queue[dict[str, Any]]"] = {}

_END = {"type": "_end", "data": {}}


def create_run() -> str:
    rid = str(uuid.uuid4())
    _runs[rid] = queue.Queue()
    return rid


def emit(run_id: Optional[str], event_type: str, data: Optional[dict] = None) -> None:
    """Publish an event to the run's queue. No-op if run_id is missing/unknown."""
    if not run_id:
        return
    q = _runs.get(run_id)
    if q is None:
        return
    q.put({"type": event_type, "data": data or {}, "ts": time.time()})


def end_run(run_id: str) -> None:
    """Signal the SSE consumer to close the stream."""
    q = _runs.get(run_id)
    if q is not None:
        q.put(_END)


def get_event(run_id: str, timeout: float = 1.0) -> Optional[dict]:
    """Block up to `timeout` seconds for the next event. None on timeout."""
    q = _runs.get(run_id)
    if q is None:
        return None
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def cleanup_run(run_id: str) -> None:
    _runs.pop(run_id, None)


def is_end(event: Optional[dict]) -> bool:
    return bool(event) and event.get("type") == "_end"
