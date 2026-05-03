"""DM conversation storage + helpers.

One row per (lead_id, channel). The ``turns`` column is a JSONB array of
``{role: 'us'|'them', message: str, ts: iso}`` — we always read the whole
thread to draft the next message, so a child table would just be more
joins for no benefit.

Channels: 'whatsapp' | 'instagram' | 'facebook'
Status:   'open' (waiting for next move)
        | 'no_response' (we sent, they ghosted)
        | 'booked' (got the call/meeting)
        | 'closed' (give up, lost, or done)

In-memory fallback mirrors the Supabase shape so dev mode without a DB
still works the same way.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

from tools.supabase_client import _memory, _memory_store_available, get_supabase

CHANNELS = ("whatsapp", "instagram", "facebook")

# Top-level memory bucket (lazy-init so we don't break old fallback shape).
if "conversations" not in _memory:
    _memory["conversations"] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_or_create(lead_id: str, channel: str) -> dict:
    """Fetch the conversation row for (lead, channel), creating it if missing."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel}")

    if _memory_store_available():
        for row in _memory["conversations"].values():
            if row["lead_id"] == lead_id and row["channel"] == channel:
                return row
        cid = str(uuid.uuid4())
        row = {
            "id": cid,
            "lead_id": lead_id,
            "channel": channel,
            "status": "open",
            "turns": [],
            "created_at": _now(),
            "updated_at": _now(),
        }
        _memory["conversations"][cid] = row
        return row

    client = get_supabase()
    try:
        existing = (
            client.table("conversations").select("*")
            .eq("lead_id", lead_id).eq("channel", channel)
            .limit(1).execute()
        )
        if existing.data:
            return existing.data[0]
        resp = client.table("conversations").insert({
            "lead_id": lead_id, "channel": channel, "status": "open", "turns": [],
        }).execute()
        return (resp.data or [{}])[0]
    except Exception as exc:
        print(f"[conversations] get_or_create failed: {exc}")
        raise


def list_for_lead(lead_id: str) -> list[dict]:
    if _memory_store_available():
        return [r for r in _memory["conversations"].values() if r["lead_id"] == lead_id]
    client = get_supabase()
    try:
        resp = (
            client.table("conversations").select("*")
            .eq("lead_id", lead_id)
            .order("created_at")
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        print(f"[conversations] list_for_lead failed: {exc}")
        return []


def append_turn(conv_id: str, role: str, message: str) -> dict:
    """Append a turn ('us' or 'them') to the conversation's JSONB turns array."""
    if role not in ("us", "them"):
        raise ValueError(f"role must be 'us' or 'them', got {role!r}")
    turn = {"role": role, "message": message, "ts": _now()}

    if _memory_store_available():
        row = _memory["conversations"].get(conv_id)
        if not row:
            raise ValueError(f"conversation {conv_id} not found")
        row["turns"].append(turn)
        row["updated_at"] = _now()
        return row

    client = get_supabase()
    try:
        cur = client.table("conversations").select("turns").eq("id", conv_id).limit(1).execute()
        turns = (cur.data or [{}])[0].get("turns") or []
        turns.append(turn)
        resp = (
            client.table("conversations")
            .update({"turns": turns, "updated_at": _now()})
            .eq("id", conv_id).execute()
        )
        return (resp.data or [{}])[0]
    except Exception as exc:
        print(f"[conversations] append_turn failed: {exc}")
        raise


def update_status(conv_id: str, status: str) -> Optional[dict]:
    if status not in ("open", "closed", "booked", "no_response"):
        raise ValueError(f"unknown status: {status}")
    if _memory_store_available():
        row = _memory["conversations"].get(conv_id)
        if not row:
            return None
        row["status"] = status
        row["updated_at"] = _now()
        return row
    client = get_supabase()
    try:
        resp = (
            client.table("conversations")
            .update({"status": status, "updated_at": _now()})
            .eq("id", conv_id).execute()
        )
        return (resp.data or [None])[0]
    except Exception as exc:
        print(f"[conversations] update_status failed: {exc}")
        return None


# ---- DM link builders ----------------------------------------------------- #
#
# These are click-to-DM links that open WhatsApp / Instagram with the
# message pre-filled. The user sends manually — that's how we stay
# compliant with platform ToS for cold outbound.


def whatsapp_link(phone: str, message: str = "") -> str:
    """Return wa.me link for a phone number with optional pre-filled text.

    ``phone`` may have spaces, dashes, parentheses — we strip them. The +
    is dropped because wa.me wants raw digits with country code.
    """
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if not digits:
        return ""
    if message:
        return f"https://wa.me/{digits}?text={quote(message)}"
    return f"https://wa.me/{digits}"


def instagram_link(username: str, message: str = "") -> str:
    """Return Instagram DM link.

    Uses ig.me/m/<username> which opens a new DM thread in the IG app or
    web. The text= param is honored on web but ignored on mobile, so we
    always also expose the message text in the dashboard for copy-paste.
    """
    u = (username or "").lstrip("@").strip()
    if not u:
        return ""
    if message:
        return f"https://ig.me/m/{u}?text={quote(message)}"
    return f"https://ig.me/m/{u}"


def facebook_link(page_url: str) -> str:
    """Best-effort: Facebook doesn't have a stable click-to-DM URL outside
    Messenger. We just return the page URL; user clicks Message on the page.
    """
    return (page_url or "").strip()
