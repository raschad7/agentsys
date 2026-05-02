"""ZeroBounce email verification client.

ZeroBounce categorises every email into one of these statuses:
    valid       — mailbox confirmed deliverable
    invalid     — mailbox confirmed undeliverable (will bounce)
    catch-all   — domain accepts all addresses; mailbox unverifiable
    unknown     — temporary lookup failure
    spamtrap    — known spam-trap address (blacklists you on send)
    abuse       — known complainer (file-a-complaint type)
    do_not_mail — opt-out / role-based / disposable

We translate that to a Verdict the Postie can act on:
    pass        — go ahead and send
    warn        — send anyway (caller decides), but mailbox isn't confirmed
    reject      — never send

Costs:
    Free tier: 100 verifications when you sign up.
    Paid: ~$0.008/verify, no monthly minimum.

Caching:
    Local SQLite at .zerobounce_cache.db, keyed by lowercased email,
    7-day TTL. Mailboxes don't change often; re-verifying is wasted spend.

Caveats:
    - We only call ZeroBounce after the cheap MX gate has already passed,
      so we don't pay to verify obviously-dead domains.
    - On API failure (network, quota), we fall back to "warn" so a flaky
      ZeroBounce doesn't block your sends — losing a deliverability
      check is better than a stalled pipeline.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

API_URL = "https://api.zerobounce.net/v2/validate"
CACHE_PATH = Path(__file__).parent.parent / ".zerobounce_cache.db"
CACHE_TTL_DAYS = 7

# Map ZB statuses to one of our 3 actionable verdicts.
_STATUS_VERDICT = {
    "valid":        "pass",
    "catch-all":    "warn",   # deliverable but unconfirmed
    "unknown":      "warn",   # transient — fail-open
    "invalid":      "reject",
    "spamtrap":     "reject",
    "abuse":        "reject",
    "do_not_mail":  "reject",
}


@dataclass
class VerificationResult:
    email: str
    status: str = ""           # raw ZB status
    sub_status: str = ""
    verdict: str = "warn"      # pass | warn | reject
    free_email: bool = False
    mx_record: str = ""
    raw: Optional[dict] = None
    cached: bool = False
    error: str = ""

    @property
    def passes(self) -> bool:
        return self.verdict == "pass"

    @property
    def rejects(self) -> bool:
        return self.verdict == "reject"


# --------------------------------------------------------------------------- #
# Cache (separate SQLite from enrichment cache)
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _conn_get() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    _conn = sqlite3.connect(str(CACHE_PATH), check_same_thread=False)
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS zb_cache (
            email      TEXT PRIMARY KEY,
            verified_at TEXT NOT NULL,
            payload    TEXT NOT NULL
        )
        """
    )
    _conn.commit()
    return _conn


def _cache_get(email: str) -> Optional[dict]:
    with _lock:
        cur = _conn_get().execute(
            "SELECT verified_at, payload FROM zb_cache WHERE email = ?",
            (email.lower(),),
        )
        row = cur.fetchone()
    if not row:
        return None
    verified_iso, payload = row
    try:
        verified_dt = datetime.fromisoformat(verified_iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if verified_dt < datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _cache_put(email: str, payload: dict) -> None:
    try:
        blob = json.dumps(payload)
    except (TypeError, ValueError):
        return
    with _lock:
        _conn_get().execute(
            "INSERT OR REPLACE INTO zb_cache (email, verified_at, payload) VALUES (?, ?, ?)",
            (email.lower(), datetime.now(timezone.utc).isoformat(), blob),
        )
        _conn_get().commit()


def cache_stats() -> dict:
    """For /api/stats — small operational summary of cached verifications."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS)).isoformat()
    with _lock:
        c = _conn_get()
        rows = c.execute("SELECT COUNT(*) FROM zb_cache").fetchone()[0]
        fresh = c.execute(
            "SELECT COUNT(*) FROM zb_cache WHERE verified_at >= ?", (cutoff,)
        ).fetchone()[0]
    return {"rows": int(rows), "fresh_rows": int(fresh), "ttl_days": CACHE_TTL_DAYS}


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def _api_key() -> str:
    return os.getenv("ZEROBOUNCE_API_KEY", "").strip()


def verify_email(email: str, *, use_cache: bool = True) -> VerificationResult:
    """Run ZeroBounce against ``email`` and return a VerificationResult.

    No-key fallback: if ZEROBOUNCE_API_KEY isn't set, returns a "warn"
    verdict with status="skipped" so the pipeline continues. This keeps
    local dev working without a key.
    """
    addr = (email or "").strip().lower()
    if not addr:
        return VerificationResult(email="", verdict="reject", status="syntax_invalid",
                                  error="empty email")

    if not _api_key():
        return VerificationResult(email=addr, verdict="warn", status="skipped",
                                  error="ZEROBOUNCE_API_KEY not set")

    if use_cache:
        cached = _cache_get(addr)
        if cached:
            r = VerificationResult(**cached)
            r.cached = True
            return r

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                API_URL,
                params={"api_key": _api_key(), "email": addr},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        # Fail-open: don't block the pipeline on a flaky ZB or quota exhaustion.
        # Surface the failure so the dashboard can show it, but verdict=warn.
        return VerificationResult(
            email=addr, verdict="warn", status="api_error",
            error=f"ZeroBounce request failed: {exc}",
        )

    status = (data.get("status") or "").lower().strip()
    verdict = _STATUS_VERDICT.get(status, "warn")
    result = VerificationResult(
        email=addr,
        status=status or "unknown",
        sub_status=str(data.get("sub_status") or ""),
        verdict=verdict,
        free_email=bool(data.get("free_email")),
        mx_record=str(data.get("mx_record") or ""),
        raw=data,
    )
    # Cache the dict-shape so we can rehydrate later
    _cache_put(addr, {
        "email": result.email,
        "status": result.status,
        "sub_status": result.sub_status,
        "verdict": result.verdict,
        "free_email": result.free_email,
        "mx_record": result.mx_record,
        "raw": result.raw,
        "error": "",
    })
    return result


def get_credit_balance() -> Optional[int]:
    """Optional: query remaining ZB credits for the dashboard footer.

    Returns None if no key set or the lookup fails.
    """
    if not _api_key():
        return None
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.get(
                "https://api.zerobounce.net/v2/getcredits",
                params={"api_key": _api_key()},
            )
            r.raise_for_status()
            data = r.json()
        # API returns {"Credits": "100"} (string). -1 means subscription tier.
        c = data.get("Credits")
        return int(c) if c is not None else None
    except Exception:
        return None
