"""Supabase client and helper functions.

If Supabase env vars are missing, the helpers fall back to an in-memory store
so the rest of the system can still be exercised end to end.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(override=True)

_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_client = None
_memory = {"leads": {}, "outreach": [], "agent_logs": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_supabase():
    """Return a Supabase client, or None when credentials are absent."""
    global _client
    if _client is not None:
        return _client
    if not _SUPABASE_URL or not _SUPABASE_KEY:
        return None
    try:
        from supabase import create_client

        _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        return _client
    except Exception as exc:  # pragma: no cover - network/driver failure
        print(f"[supabase] init failed, using in-memory store: {exc}")
        return None


def _memory_store_available() -> bool:
    return get_supabase() is None


def insert_lead(lead: dict) -> dict:
    """Insert a lead and return the stored row (with id).

    Normalises empty strings on the unique-constrained ``email`` column to
    NULL — PostgreSQL allows multiple NULLs in a UNIQUE column but rejects
    duplicate empty strings (so an OSM lead with no email would collide
    with the next OSM lead that also has no email). Same logic for any
    other identifier-shaped fields where "" is meaningless.
    """
    lead = {k: v for k, v in lead.items() if v is not None}
    if lead.get("email") == "":
        lead.pop("email", None)  # drop → NULL on insert
    if _memory_store_available():
        lid = lead.get("id") or str(uuid.uuid4())
        row = {
            "id": lid,
            "name": lead.get("name"),
            "email": lead.get("email"),
            "company": lead.get("company"),
            "website": lead.get("website"),
            "industry": lead.get("industry"),
            "instagram": lead.get("instagram", ""),
            "facebook": lead.get("facebook", ""),
            "phone": lead.get("phone", ""),
            "source": lead.get("source", ""),
            "score": lead.get("score", 0),
            "status": lead.get("status", "new"),
            "created_at": _now(),
        }
        _memory["leads"][lid] = row
        return row
    client = get_supabase()
    try:
        resp = client.table("leads").insert(lead).execute()
        data = resp.data or []
        return data[0] if data else lead
    except Exception as exc:
        # Upsert-by-email fallback for duplicate emails
        try:
            if lead.get("email"):
                existing = (
                    client.table("leads")
                    .select("*")
                    .eq("email", lead["email"])
                    .limit(1)
                    .execute()
                )
                if existing.data:
                    return existing.data[0]
        except Exception:
            pass
        print(f"[supabase] insert_lead failed: {exc}")
        raise


def get_lead(lead_id: str) -> Optional[dict]:
    if _memory_store_available():
        return _memory["leads"].get(lead_id)
    client = get_supabase()
    try:
        resp = client.table("leads").select("*").eq("id", lead_id).limit(1).execute()
        return resp.data[0] if resp.data else None
    except Exception as exc:
        print(f"[supabase] get_lead failed: {exc}")
        return None


def update_lead(lead_id: str, data: dict) -> Optional[dict]:
    if _memory_store_available():
        row = _memory["leads"].get(lead_id)
        if not row:
            return None
        row.update(data)
        return row
    client = get_supabase()
    try:
        resp = client.table("leads").update(data).eq("id", lead_id).execute()
        return (resp.data or [None])[0]
    except Exception as exc:
        print(f"[supabase] update_lead failed: {exc}")
        return None


def insert_outreach(lead_id: str, subject: str, body: str, status: str = "sent") -> dict:
    row = {
        "lead_id": lead_id,
        "email_subject": subject,
        "email_body": body,
        "sent_at": _now() if status == "sent" else None,
        "status": status,
    }
    if _memory_store_available():
        row = {"id": str(uuid.uuid4()), **row, "opened": False, "replied": False, "follow_up_count": 0}
        _memory["outreach"].append(row)
        return row
    client = get_supabase()
    try:
        resp = client.table("outreach").insert(row).execute()
        return (resp.data or [row])[0]
    except Exception as exc:
        print(f"[supabase] insert_outreach failed: {exc}")
        return row


def update_outreach_by_lead(lead_id: str, data: dict) -> Optional[dict]:
    """Update the most recent outreach row for a given lead.

    When status flips to 'sent', auto-stamp sent_at if the caller didn't
    pass one explicitly — that way the outreach agent never has to
    remember to do it.
    """
    payload = dict(data)
    if payload.get("status") == "sent" and "sent_at" not in payload:
        payload["sent_at"] = _now()

    if _memory_store_available():
        rows = [r for r in _memory["outreach"] if r["lead_id"] == lead_id]
        if not rows:
            return None
        row = rows[-1]
        row.update(payload)
        return row
    client = get_supabase()
    try:
        resp = (
            client.table("outreach")
            .update(payload)
            .eq("lead_id", lead_id)
            .execute()
        )
        return (resp.data or [None])[0]
    except Exception as exc:
        print(f"[supabase] update_outreach_by_lead failed: {exc}")
        return None


def log_action(
    agent_name: str,
    lead_id: Optional[str],
    action: str,
    result: str = "",
    error: str = "",
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> None:
    row: dict[str, Any] = {
        "agent_name": agent_name,
        "lead_id": lead_id,
        "action": action,
        "result": result,
        "error": error,
    }
    if model is not None:
        row["model"] = model
    if prompt_tokens is not None:
        row["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        row["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        row["total_tokens"] = total_tokens

    if _memory_store_available():
        _memory["agent_logs"].append({"id": str(uuid.uuid4()), **row, "created_at": _now()})
        return
    client = get_supabase()
    try:
        client.table("agent_logs").insert(row).execute()
    except Exception as exc:
        print(f"[supabase] log_action failed: {exc}")


def get_all_leads(status: Optional[str] = None) -> list[dict]:
    if _memory_store_available():
        rows = list(_memory["leads"].values())
        if status:
            rows = [r for r in rows if r.get("status") == status]
        return rows
    client = get_supabase()
    try:
        q = client.table("leads").select("*")
        if status:
            q = q.eq("status", status)
        resp = q.execute()
        return resp.data or []
    except Exception as exc:
        print(f"[supabase] get_all_leads failed: {exc}")
        return []


def get_recent_logs(limit: int = 50) -> list[dict]:
    if _memory_store_available():
        return sorted(
            _memory["agent_logs"], key=lambda r: r.get("created_at", ""), reverse=True
        )[:limit]
    client = get_supabase()
    try:
        resp = (
            client.table("agent_logs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        print(f"[supabase] get_recent_logs failed: {exc}")
        return []


def _domain_of(value: str) -> str:
    """Lowercased registrable host from email or url. Empty if can't extract."""
    if not value:
        return ""
    s = value.strip().lower()
    if "@" in s and "://" not in s:
        s = s.split("@", 1)[1]
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0].split("?", 1)[0]
    if s.startswith("www."):
        s = s[4:]
    return s


def lead_exists_for_domain(domain: str, within_days: int = 90) -> Optional[dict]:
    """True if any lead in the last ``within_days`` shares this domain
    (matched against email host or website host).

    Returns the matching row (handy for telling the user "we already saw this
    one on 2026-04-15") or None.
    """
    if not domain:
        return None
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=int(within_days))
    cutoff = cutoff_dt.isoformat()

    if _memory_store_available():
        for row in _memory["leads"].values():
            if (row.get("created_at") or "") < cutoff:
                continue
            if _domain_of(row.get("email") or "") == domain:
                return row
            if _domain_of(row.get("website") or "") == domain:
                return row
        return None
    client = get_supabase()
    try:
        # Two cheap lookups (one by email, one by website) — both indexed.
        email_like = f"%@{domain}"
        site_like1 = f"%//{domain}%"
        site_like2 = f"%//{domain}/%"
        q1 = (
            client.table("leads").select("*")
            .gte("created_at", cutoff)
            .ilike("email", email_like)
            .limit(1).execute()
        )
        if q1.data:
            return q1.data[0]
        for pattern in (site_like1, site_like2):
            q2 = (
                client.table("leads").select("*")
                .gte("created_at", cutoff)
                .ilike("website", pattern)
                .limit(1).execute()
            )
            if q2.data:
                return q2.data[0]
        return None
    except Exception as exc:
        print(f"[supabase] lead_exists_for_domain failed: {exc}")
        return None


def get_latest_outreach_for_lead(lead_id: str) -> Optional[dict]:
    """Return the most recent outreach row for ``lead_id`` (any status)."""
    if not lead_id:
        return None
    if _memory_store_available():
        rows = [r for r in _memory["outreach"] if r["lead_id"] == lead_id]
        return rows[-1] if rows else None
    client = get_supabase()
    try:
        resp = (
            client.table("outreach")
            .select("*")
            .eq("lead_id", lead_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return (resp.data or [None])[0]
    except Exception as exc:
        print(f"[supabase] get_latest_outreach_for_lead failed: {exc}")
        return None


def find_outreach_by_email(email: str) -> Optional[dict]:
    """Look up the most recent outreach row via the lead's email."""
    if _memory_store_available():
        lead = next(
            (l for l in _memory["leads"].values() if l.get("email") == email), None
        )
        if not lead:
            return None
        rows = [r for r in _memory["outreach"] if r["lead_id"] == lead["id"]]
        return rows[-1] if rows else None
    client = get_supabase()
    try:
        lead_resp = (
            client.table("leads").select("id").eq("email", email).limit(1).execute()
        )
        if not lead_resp.data:
            return None
        lead_id = lead_resp.data[0]["id"]
        out_resp = (
            client.table("outreach")
            .select("*")
            .eq("lead_id", lead_id)
            .order("sent_at", desc=True)
            .limit(1)
            .execute()
        )
        return (out_resp.data or [None])[0]
    except Exception as exc:
        print(f"[supabase] find_outreach_by_email failed: {exc}")
        return None
