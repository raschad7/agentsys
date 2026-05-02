"""Local SQLite-backed cache for enrichment results.

Purpose:
    Re-running searches over the same niche/region surfaces the same
    websites repeatedly. Without a cache we'd re-fetch each one — slow
    and rude to the target servers. SQLite is enough at our scale,
    survives restarts, and costs nothing.

Key:
    Lowercased registrable host (e.g. "example.com"). Two leads sharing a
    domain share enrichment.

TTL:
    24 hours by default. Tunable via ENRICHMENT_CACHE_TTL_HOURS env.

Disable:
    ENRICHMENT_CACHE=off in .env.

Caveats:
    - We cache the ENTIRE evidence dict, including signals derived from
      lead-supplied metadata (e.g. social_silent which depends on
      latest_post_iso). The caller is expected to recompute those bits
      after a cache hit so they reflect the current lead's data.
    - The cache file lives at the project root (.enrichment_cache.db)
      and is excluded from git via .gitignore convention.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

CACHE_PATH = Path(__file__).parent.parent / ".enrichment_cache.db"
DEFAULT_TTL_HOURS = 24

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _ttl_hours() -> int:
    raw = os.getenv("ENRICHMENT_CACHE_TTL_HOURS", "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_TTL_HOURS
    except ValueError:
        return DEFAULT_TTL_HOURS


def _enabled() -> bool:
    return os.getenv("ENRICHMENT_CACHE", "on").strip().lower() not in ("off", "0", "false")


def _conn_get() -> sqlite3.Connection:
    """Lazy-open the SQLite connection. Single connection, mutex-guarded."""
    global _conn
    if _conn is not None:
        return _conn
    _conn = sqlite3.connect(str(CACHE_PATH), check_same_thread=False)
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_cache (
            domain     TEXT PRIMARY KEY,
            fetched_at TEXT NOT NULL,
            payload    TEXT NOT NULL
        )
        """
    )
    _conn.commit()
    return _conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get(domain: str) -> Optional[dict[str, Any]]:
    """Return cached evidence for ``domain`` if fresh, else None."""
    if not _enabled() or not domain:
        return None
    key = domain.lower().strip()
    with _lock:
        cur = _conn_get().execute(
            "SELECT fetched_at, payload FROM enrichment_cache WHERE domain = ?",
            (key,),
        )
        row = cur.fetchone()
    if not row:
        return None
    fetched_iso, payload = row
    try:
        fetched_dt = datetime.fromisoformat(fetched_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_ttl_hours())
    if fetched_dt < cutoff:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def put(domain: str, evidence: dict[str, Any]) -> None:
    """Cache ``evidence`` under ``domain`` (overwrites prior entry)."""
    if not _enabled() or not domain or not evidence:
        return
    try:
        payload = json.dumps(evidence, default=str)
    except (TypeError, ValueError):
        return
    key = domain.lower().strip()
    with _lock:
        _conn_get().execute(
            "INSERT OR REPLACE INTO enrichment_cache (domain, fetched_at, payload) VALUES (?, ?, ?)",
            (key, _now_iso(), payload),
        )
        _conn_get().commit()


def stats() -> dict[str, Any]:
    """Operational stats for the /api/stats endpoint."""
    if not _enabled():
        return {"enabled": False, "rows": 0, "ttl_hours": _ttl_hours()}
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=_ttl_hours())).isoformat()
    with _lock:
        c = _conn_get()
        rows = c.execute("SELECT COUNT(*) FROM enrichment_cache").fetchone()[0]
        fresh = c.execute(
            "SELECT COUNT(*) FROM enrichment_cache WHERE fetched_at >= ?",
            (cutoff_iso,),
        ).fetchone()[0]
    return {
        "enabled": True,
        "rows": int(rows),
        "fresh_rows": int(fresh),
        "ttl_hours": _ttl_hours(),
    }


def clear() -> int:
    """Wipe the cache. Returns number of rows removed."""
    with _lock:
        c = _conn_get()
        n = c.execute("SELECT COUNT(*) FROM enrichment_cache").fetchone()[0]
        c.execute("DELETE FROM enrichment_cache")
        c.commit()
    return int(n)
