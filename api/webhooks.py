"""FastAPI application: dashboard, pipeline endpoints, and live SSE stream."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import secrets
import sys
import threading
from pathlib import Path
from typing import Any, Optional

# When uvicorn is launched directly (uvicorn api.webhooks:app), main.py's
# stdout reconfigure never runs. Repeat it here so prints with arrows, bullets,
# or Arabic data don't crash on Windows cp1252/cp1256 stdout.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agents.dm_agent import draft_opener as dm_draft_opener, draft_reply as dm_draft_reply
from agents.outreach import draft_email_for_lead, send_existing_draft
from graph.pipeline import run_pipeline
from graph.search_pipeline import run_search
from tools import campaigns as campaigns_tool
from tools import conversations as conv_tool
from tools import enrichment_cache, events, llm
from tools.supabase_client import (
    find_outreach_by_email,
    get_all_leads,
    get_latest_outreach_for_lead,
    get_lead,
    get_recent_logs,
    insert_lead,
    update_lead,
    update_outreach_by_lead,
)

app = FastAPI(title="AgentFlow")

WEB_DIR = Path(__file__).parent.parent / "web"


# ---- API key auth -------------------------------------------------------- #
#
# Tier-A1: stop the dashboard's destructive endpoints from being open to
# anyone on the LAN (uvicorn binds 0.0.0.0). The key lives in env (preferred)
# or is auto-generated at process start (printed once on stdout for the
# operator to copy). The dashboard reads it from a <meta> tag injected at
# render time so the JS can attach `X-API-Key` to every /api/ call.

_AUTH_KEY = os.getenv("AGENTFLOW_API_KEY", "").strip()
if not _AUTH_KEY:
    _AUTH_KEY = secrets.token_urlsafe(24)
    print(f"[auth] AGENTFLOW_API_KEY not set — generated session key: {_AUTH_KEY}")
    print("[auth]   set AGENTFLOW_API_KEY in .env to keep it stable across restarts")
else:
    print(f"[auth] AGENTFLOW_API_KEY loaded ({len(_AUTH_KEY)} chars)")


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: 401 if X-API-Key header doesn't match."""
    if not x_api_key or not secrets.compare_digest(x_api_key, _AUTH_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


# ---- Static dashboard ----------------------------------------------------- #


@app.get("/")
def dashboard() -> HTMLResponse:
    """Serve the dashboard with the API key injected as a <meta> tag.

    The dashboard JS reads ``<meta name="agentflow-api-key">`` and includes
    it on every fetch as ``X-API-Key``. This means anyone who can load the
    HTML can use the dashboard — which is fine, because exposing the dashboard
    URL is itself the threat we're protecting against (LAN attackers).
    """
    raw = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    inject = f'  <meta name="agentflow-api-key" content="{_AUTH_KEY}" />\n'
    if "<head>" in raw:
        raw = raw.replace("<head>", "<head>\n" + inject, 1)
    return HTMLResponse(raw)


# ---- Read-only API -------------------------------------------------------- #


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "running"}


@app.get("/api/leads/export", dependencies=[Depends(require_api_key)])
def export_leads(status: Optional[str] = None):
    """Download all leads (or a filtered subset) as a CSV file.

    Query param ``status`` filters by lead status, e.g.
    ``/api/leads/export?status=qualified``
    """
    rows = get_all_leads(status=status)
    cols = ["id", "company", "name", "email", "website", "phone",
            "instagram", "facebook", "industry", "score", "status",
            "source", "created_at"]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: (row.get(c) or "") for c in cols})

    filename = f"leads{'_' + status if status else ''}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/leads")
def list_leads() -> list[dict[str, Any]]:
    rows = get_all_leads()
    return [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "email": r.get("email"),
            "company": r.get("company"),
            "score": r.get("score", 0),
            "status": r.get("status", "new"),
        }
        for r in rows
    ]


@app.get("/logs")
def list_logs() -> list[dict[str, Any]]:
    return get_recent_logs(limit=50)


@app.get("/api/stats", dependencies=[Depends(require_api_key)])
def api_stats() -> dict[str, Any]:
    """Operational visibility — drives the dashboard footer.

    Returns:
      openai:  today's spend + cap (in-memory, resets at process restart)
      leads:   total + breakdown by status + breakdown by source
      cache:   enrichment cache rows + freshness count + ttl
    """
    rows = get_all_leads()
    by_status: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in rows:
        s = (r.get("status") or "?") or "?"
        by_status[s] = by_status.get(s, 0) + 1
        src = (r.get("source") or "(unknown)") or "(unknown)"
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "openai": llm.get_daily_spend(),
        "leads": {
            "total": len(rows),
            "by_status": by_status,
            "by_source": by_source,
        },
        "cache": enrichment_cache.stats(),
    }


# ---- Pipeline runners ----------------------------------------------------- #


class QuickRunBody(BaseModel):
    name: Optional[str] = ""
    email: Optional[str] = ""
    company: Optional[str] = ""
    website: Optional[str] = ""
    industry: Optional[str] = ""


class SearchBody(BaseModel):
    location: str
    niche: str
    count: int = 5
    # B11 — optional campaign id; null = use default global rubric
    campaign_id: Optional[str] = None


class CampaignBody(BaseModel):
    name: str
    description: Optional[str] = ""
    pitch_angle: Optional[str] = ""
    signal_weights: Optional[dict[str, Any]] = None
    score_threshold: Optional[int] = 3
    active: Optional[bool] = True


class CampaignPatchBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    pitch_angle: Optional[str] = None
    signal_weights: Optional[dict[str, Any]] = None
    score_threshold: Optional[int] = None
    active: Optional[bool] = None


def _start_run_for_lead(lead: dict) -> str:
    """Spawn the pipeline in a background thread and return a run_id."""
    run_id = events.create_run()

    def _runner():
        try:
            run_pipeline(lead, run_id=run_id)
        finally:
            events.end_run(run_id)

    threading.Thread(target=_runner, daemon=True).start()
    return run_id


@app.post("/api/search", dependencies=[Depends(require_api_key)])
def api_search(body: SearchBody) -> dict[str, str]:
    """Kick off a multi-lead discovery + processing run."""
    if not body.location.strip() or not body.niche.strip():
        raise HTTPException(status_code=400, detail="location and niche are required")
    count = max(1, min(int(body.count or 5), 20))

    # B11: validate campaign_id if supplied — fail fast rather than silently
    # ignoring a bad id and using global defaults.
    if body.campaign_id:
        campaign = campaigns_tool.get_campaign(body.campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="campaign not found")
        if not campaign.get("active", True):
            raise HTTPException(status_code=400, detail="campaign is inactive")

    run_id = events.create_run()

    def _runner():
        try:
            run_search(
                body.location.strip(),
                body.niche.strip(),
                count,
                run_id=run_id,
                campaign_id=body.campaign_id or None,
            )
        finally:
            events.end_run(run_id)

    threading.Thread(target=_runner, daemon=True).start()
    return {"run_id": run_id}


# ---- Campaigns CRUD (B11) ------------------------------------------------ #


@app.get("/api/campaigns", dependencies=[Depends(require_api_key)])
def api_campaigns_list(active: bool = False) -> list[dict[str, Any]]:
    return campaigns_tool.list_campaigns(only_active=active)


@app.post("/api/campaigns", dependencies=[Depends(require_api_key)])
def api_campaigns_create(body: CampaignBody) -> dict[str, Any]:
    try:
        return campaigns_tool.create_campaign(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.patch("/api/campaigns/{campaign_id}", dependencies=[Depends(require_api_key)])
def api_campaigns_update(campaign_id: str, body: CampaignPatchBody) -> dict[str, Any]:
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        row = campaigns_tool.update_campaign(campaign_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not row:
        raise HTTPException(status_code=404, detail="campaign not found")
    return row


@app.delete("/api/campaigns/{campaign_id}", dependencies=[Depends(require_api_key)])
def api_campaigns_deactivate(campaign_id: str) -> dict[str, Any]:
    """Soft-delete by flipping active=false (preserves historical leads' FK)."""
    row = campaigns_tool.deactivate_campaign(campaign_id)
    if not row:
        raise HTTPException(status_code=404, detail="campaign not found")
    return row


@app.post("/api/run", dependencies=[Depends(require_api_key)])
def api_run(body: QuickRunBody) -> dict[str, str]:
    """Insert a fresh lead from form data and start the pipeline."""
    payload = {k: v for k, v in body.model_dump().items() if v}
    if not payload.get("email"):
        raise HTTPException(status_code=400, detail="email is required")
    lead = insert_lead(payload)
    run_id = _start_run_for_lead(lead)
    return {"run_id": run_id, "lead_id": lead.get("id", "")}


@app.post("/api/run/{lead_id}", dependencies=[Depends(require_api_key)])
def api_run_existing(lead_id: str) -> dict[str, str]:
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    run_id = _start_run_for_lead(lead)
    return {"run_id": run_id, "lead_id": lead_id}


# ---- Manual-control endpoints (Scribe + Postie) -------------------------- #


class DraftEditBody(BaseModel):
    subject: str
    body: str


@app.post("/api/draft/{lead_id}", dependencies=[Depends(require_api_key)])
def api_draft(lead_id: str) -> dict[str, Any]:
    """✍️ Scribe writes (or re-writes) the cold email for one lead.

    Saves draft as a 'pending' outreach row and flips lead status to 'drafted'.
    Returns the subject + body so the dashboard can show it immediately.
    """
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    if lead.get("status") not in ("qualified", "drafted"):
        raise HTTPException(
            status_code=400,
            detail=f"lead must be 'qualified' (or already 'drafted'); got {lead.get('status')!r}",
        )
    result = draft_email_for_lead(lead)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return {
        "lead_id": lead_id,
        "subject": result["subject"],
        "body": result["body"],
        "status": "drafted",
    }


@app.post("/api/edit/{lead_id}", dependencies=[Depends(require_api_key)])
def api_edit_draft(lead_id: str, body: DraftEditBody) -> dict[str, Any]:
    """Edit the saved draft (subject and/or body) before sending."""
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    if not body.subject.strip() or not body.body.strip():
        raise HTTPException(status_code=400, detail="subject and body are required")
    out = get_latest_outreach_for_lead(lead_id)
    if not out:
        raise HTTPException(status_code=404, detail="no draft to edit — run Scribe first")
    update_outreach_by_lead(
        lead_id,
        {"email_subject": body.subject.strip(), "email_body": body.body.strip()},
    )
    return {"lead_id": lead_id, "subject": body.subject, "body": body.body, "status": "drafted"}


@app.post("/api/send/{lead_id}", dependencies=[Depends(require_api_key)])
def api_send(lead_id: str) -> dict[str, Any]:
    """📮 Postie sends the existing draft via Instantly."""
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    if lead.get("status") != "drafted":
        raise HTTPException(
            status_code=400,
            detail=f"lead must be 'drafted'; got {lead.get('status')!r}",
        )
    result = send_existing_draft(lead)
    if not result["success"]:
        # Don't 500 — the dashboard wants to display the error on the card
        return {
            "lead_id": lead_id,
            "success": False,
            "status": "error",
            "error": result["error"],
        }
    return {
        "lead_id": lead_id,
        "success": True,
        "status": "contacted",
        "mode": result["mode"],
        "subject": result["subject"],
    }


@app.post("/api/reject/{lead_id}", dependencies=[Depends(require_api_key)])
def api_reject(lead_id: str) -> dict[str, Any]:
    """Manually reject a lead (skip Scribe + Postie entirely)."""
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    update_lead(lead_id, {"status": "rejected"})
    return {"lead_id": lead_id, "status": "rejected"}


# ---- DM conversations (WhatsApp / Instagram / Facebook) ------------------ #
#
# Cold-DMing through WhatsApp Business / Instagram Messenger APIs is
# blocked by Meta's policies. We work around that with click-to-DM links:
# the dashboard generates a wa.me / ig.me URL with a personalized opener
# pre-filled, the user clicks Send manually in the app, then comes back
# and pastes the reply. The dm_agent handles the conversational drafting.


def _conv_payload(conv: dict, lead: dict) -> dict:
    """Bundle conv + the helper links the dashboard needs to render it."""
    channel = conv.get("channel", "")
    last_us = next(
        (t["message"] for t in reversed(conv.get("turns") or []) if t.get("role") == "us"),
        "",
    )
    # The link is pre-filled with the most recent OUR message so the user
    # can click straight to the app and send it without copy-pasting.
    if channel == "whatsapp":
        link = conv_tool.whatsapp_link(lead.get("phone", ""), last_us)
    elif channel == "instagram":
        link = conv_tool.instagram_link(lead.get("instagram", ""), last_us)
    elif channel == "facebook":
        link = conv_tool.facebook_link(lead.get("facebook", ""))
    else:
        link = ""
    return {
        "id": conv.get("id"),
        "lead_id": conv.get("lead_id"),
        "channel": channel,
        "status": conv.get("status"),
        "turns": conv.get("turns") or [],
        "link": link,
        "updated_at": conv.get("updated_at"),
    }


class DMDraftBody(BaseModel):
    channel: str  # whatsapp | instagram | facebook


class DMTurnBody(BaseModel):
    channel: str
    role: str  # us | them
    message: str


@app.get("/api/conversation/{lead_id}", dependencies=[Depends(require_api_key)])
def api_conversation_list(lead_id: str) -> dict[str, Any]:
    """List all DM conversations for a lead (one per channel)."""
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    convs = conv_tool.list_for_lead(lead_id)
    return {
        "lead": {
            "id": lead.get("id"),
            "company": lead.get("company"),
            "phone": lead.get("phone"),
            "instagram": lead.get("instagram"),
            "facebook": lead.get("facebook"),
        },
        "conversations": [_conv_payload(c, lead) for c in convs],
    }


@app.post("/api/conversation/{lead_id}/draft", dependencies=[Depends(require_api_key)])
def api_conversation_draft(lead_id: str, body: DMDraftBody) -> dict[str, Any]:
    """Draft the next DM message.

    If the conversation has no turns yet → draft an opener.
    Otherwise → draft a reply based on full history (and the most recent
    'them' message if present).

    The drafted message is NOT auto-appended; the user reviews it, sends
    it manually in WhatsApp/IG, then comes back and POSTs to /turn with
    role='us' to log that it went out. This keeps the log honest.
    """
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    if body.channel not in conv_tool.CHANNELS:
        raise HTTPException(status_code=400, detail=f"channel must be one of {conv_tool.CHANNELS}")

    conv = conv_tool.get_or_create(lead_id, body.channel)
    turns = conv.get("turns") or []

    if not turns:
        result = dm_draft_opener(lead, body.channel)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {
            "lead_id": lead_id,
            "channel": body.channel,
            "type": "opener",
            "message": result["message"],
            "intent": "opener",
        }

    # Find their last message to highlight it for the model.
    last_them = next(
        (t["message"] for t in reversed(turns) if t.get("role") == "them"),
        None,
    )
    result = dm_draft_reply(lead, body.channel, turns, last_them)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return {
        "lead_id": lead_id,
        "channel": body.channel,
        "type": "reply",
        "message": result["message"],
        "intent": result.get("intent", "continue"),
    }


@app.post("/api/conversation/{lead_id}/turn", dependencies=[Depends(require_api_key)])
def api_conversation_turn(lead_id: str, body: DMTurnBody) -> dict[str, Any]:
    """Log a turn (either 'us' that we just sent, or 'them' that came back).

    Side effects:
      - first 'us' turn flips lead status from 'no_contact_email' to 'dming'
      - logging a 'them' turn keeps lead status as 'dming' (or sets it back
        from 'no_response' if we'd given up on them earlier)
    """
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    if body.channel not in conv_tool.CHANNELS:
        raise HTTPException(status_code=400, detail=f"channel must be one of {conv_tool.CHANNELS}")
    if body.role not in ("us", "them"):
        raise HTTPException(status_code=400, detail="role must be 'us' or 'them'")
    msg = (body.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is empty")

    conv = conv_tool.get_or_create(lead_id, body.channel)
    conv = conv_tool.append_turn(conv["id"], body.role, msg)

    cur_status = lead.get("status")
    if cur_status in ("no_contact_email", "no_response", "dming"):
        update_lead(lead_id, {"status": "dming"})

    return {
        "lead_id": lead_id,
        "conversation": _conv_payload(conv, lead),
    }


class DMStatusBody(BaseModel):
    channel: str
    status: str  # open | closed | booked | no_response


@app.post("/api/conversation/{lead_id}/status", dependencies=[Depends(require_api_key)])
def api_conversation_status(lead_id: str, body: DMStatusBody) -> dict[str, Any]:
    """Update conversation status (booked / closed / no_response)."""
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    conv = conv_tool.get_or_create(lead_id, body.channel)
    conv = conv_tool.update_status(conv["id"], body.status) or conv
    # Mirror booked/closed back onto the lead so the dashboard counters reflect it.
    if body.status == "booked":
        update_lead(lead_id, {"status": "contacted"})
    elif body.status == "closed":
        update_lead(lead_id, {"status": "rejected"})
    return {"lead_id": lead_id, "conversation": _conv_payload(conv, lead)}


@app.get("/api/lead/{lead_id}", dependencies=[Depends(require_api_key)])
def api_get_lead(lead_id: str) -> dict[str, Any]:
    """Fetch full lead state including the latest draft (for dashboard refresh)."""
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    out = get_latest_outreach_for_lead(lead_id)
    convs = conv_tool.list_for_lead(lead_id)
    return {
        "lead": lead,
        "draft": {
            "subject": (out or {}).get("email_subject", ""),
            "body": (out or {}).get("email_body", ""),
            "status": (out or {}).get("status", ""),
            "sent_at": (out or {}).get("sent_at"),
        } if out else None,
        "conversations": [_conv_payload(c, lead) for c in convs],
    }


# ---- Legacy synchronous one-shot run (still used by tests + /import) ----- #


@app.post("/run/{lead_id}")
def run_lead_sync(lead_id: str) -> dict[str, Any]:
    """Synchronous version (used by CLI / tests). Returns final state."""
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(status_code=404, detail="lead not found")
    final = run_pipeline(lead)
    return {
        "lead_id": lead_id,
        "score": final.get("score"),
        "status": final.get("status"),
        "outreach_sent": final.get("outreach_sent", False),
        "email_subject": final.get("email_subject", ""),
        "error": final.get("error", ""),
    }


@app.post("/import", dependencies=[Depends(require_api_key)])
async def import_leads(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="body must be a JSON array")

    qualified = 0
    rejected = 0
    sent = 0
    errors = 0
    for item in payload:
        if not isinstance(item, dict):
            errors += 1
            continue
        row = insert_lead(item)
        final = run_pipeline(row)
        status = final.get("status")
        if status in ("qualified", "contacted"):
            qualified += 1
        elif status == "rejected":
            rejected += 1
        if final.get("outreach_sent"):
            sent += 1
        if final.get("error"):
            errors += 1

    return {
        "imported": len(payload),
        "qualified": qualified,
        "rejected": rejected,
        "emails_sent": sent,
        "errors": errors,
    }


# ---- Reply webhook -------------------------------------------------------- #


@app.post("/webhook/reply")
async def webhook_reply(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    email = (
        payload.get("lead_email")
        or payload.get("email")
        or payload.get("to")
        or payload.get("from")
        or ""
    )
    if not email:
        raise HTTPException(status_code=400, detail="reply payload missing email")

    out = find_outreach_by_email(email)
    if not out:
        raise HTTPException(status_code=404, detail="no outreach found for email")

    update_outreach_by_lead(out["lead_id"], {"replied": True, "status": "replied"})
    update_lead(out["lead_id"], {"status": "replied"})
    return {"ok": True, "lead_id": out["lead_id"]}


# ---- Live event stream (Server-Sent Events) ------------------------------- #


@app.get("/api/stream/{run_id}")
async def stream(run_id: str):
    """SSE stream of agent events for a single run."""

    async def gen():
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, events.get_event, run_id, 1.0)
            if event is None:
                # heartbeat — keeps proxies / browsers from closing the stream
                yield ": ping\n\n"
                continue
            if events.is_end(event):
                yield "event: end\ndata: {}\n\n"
                events.cleanup_run(run_id)
                return
            yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
