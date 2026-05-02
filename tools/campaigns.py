"""Campaigns / ICP layer (Tier-B11).

A campaign bundles three things that change per ICP:
    1. ``signal_weights`` — partial override of SIGNAL_WEIGHTS in the scorer.
       Example: a campaign aimed at e-commerce might weight ``slow_site``
       at +2 (perf is conversion) instead of the default +1.
    2. ``score_threshold`` — what counts as "qualified" for this audience.
    3. ``description`` + ``pitch_angle`` — text the Scribe injects into the
       outreach prompt so emails sound right for this audience.

If a search runs without a campaign_id, the system uses defaults (the
existing global SIGNAL_WEIGHTS + threshold=3 + the original prompt). That
keeps the original flow working — campaigns are an opt-in enhancement.

Falls back to in-memory storage when Supabase isn't reachable, like the
rest of the project.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from tools.supabase_client import _memory, _memory_store_available, get_supabase

# Memory tier
_memory.setdefault("campaigns", {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client():
    return get_supabase()


def _validate(payload: dict) -> dict:
    """Coerce + validate the campaign dict before insert/update."""
    name = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("campaign name is required")
    weights = payload.get("signal_weights")
    if weights is not None and not isinstance(weights, dict):
        raise ValueError("signal_weights must be an object")
    threshold = payload.get("score_threshold", 3)
    try:
        threshold = max(0, min(5, int(threshold)))
    except (TypeError, ValueError):
        threshold = 3
    return {
        "name": name,
        "description": (payload.get("description") or "").strip() or None,
        "pitch_angle": (payload.get("pitch_angle") or "").strip() or None,
        "signal_weights": weights,
        "score_threshold": threshold,
        "active": bool(payload.get("active", True)),
    }


def list_campaigns(*, only_active: bool = False) -> list[dict]:
    if _memory_store_available():
        rows = list(_memory["campaigns"].values())
        if only_active:
            rows = [r for r in rows if r.get("active", True)]
        return sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)
    try:
        q = _client().table("campaigns").select("*").order("created_at", desc=True)
        if only_active:
            q = q.eq("active", True)
        return q.execute().data or []
    except Exception as exc:
        print(f"[campaigns] list failed: {exc}")
        return []


def get_campaign(campaign_id: str) -> Optional[dict]:
    if not campaign_id:
        return None
    if _memory_store_available():
        return _memory["campaigns"].get(campaign_id)
    try:
        resp = (
            _client().table("campaigns").select("*").eq("id", campaign_id).limit(1).execute()
        )
        return (resp.data or [None])[0]
    except Exception as exc:
        print(f"[campaigns] get failed: {exc}")
        return None


def create_campaign(payload: dict) -> dict:
    row = _validate(payload)
    if _memory_store_available():
        cid = str(uuid.uuid4())
        full = {"id": cid, **row, "created_at": _now(), "updated_at": _now()}
        _memory["campaigns"][cid] = full
        return full
    try:
        resp = _client().table("campaigns").insert(row).execute()
        return (resp.data or [row])[0]
    except Exception as exc:
        print(f"[campaigns] create failed: {exc}")
        raise


def update_campaign(campaign_id: str, payload: dict) -> Optional[dict]:
    # Only validate keys the caller actually sent (partial update)
    patch: dict[str, Any] = {}
    if "name" in payload:
        n = (payload.get("name") or "").strip()
        if not n:
            raise ValueError("name cannot be empty")
        patch["name"] = n
    if "description" in payload:
        patch["description"] = (payload.get("description") or "").strip() or None
    if "pitch_angle" in payload:
        patch["pitch_angle"] = (payload.get("pitch_angle") or "").strip() or None
    if "signal_weights" in payload:
        w = payload.get("signal_weights")
        if w is not None and not isinstance(w, dict):
            raise ValueError("signal_weights must be an object")
        patch["signal_weights"] = w
    if "score_threshold" in payload:
        try:
            patch["score_threshold"] = max(0, min(5, int(payload["score_threshold"])))
        except (TypeError, ValueError):
            patch["score_threshold"] = 3
    if "active" in payload:
        patch["active"] = bool(payload["active"])
    if not patch:
        return get_campaign(campaign_id)

    patch["updated_at"] = _now()

    if _memory_store_available():
        row = _memory["campaigns"].get(campaign_id)
        if not row:
            return None
        row.update(patch)
        return row
    try:
        resp = _client().table("campaigns").update(patch).eq("id", campaign_id).execute()
        return (resp.data or [None])[0]
    except Exception as exc:
        print(f"[campaigns] update failed: {exc}")
        return None


def deactivate_campaign(campaign_id: str) -> Optional[dict]:
    """Soft-delete: flip active=false. We never hard-delete because leads
    may FK-reference the campaign and we want the historical record."""
    return update_campaign(campaign_id, {"active": False})
