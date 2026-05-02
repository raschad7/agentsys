"""Outreach agents — Scribe (drafts) and Postie (sends).

Two distinct functions reflecting the human-in-the-loop dashboard buttons:

    draft_email_for_lead(lead)      — Scribe ✍️  : generate subject+body,
                                                  insert outreach row as
                                                  pending, mark lead "drafted"
    send_existing_draft(lead)       — Postie 📮 : pick up the latest draft
                                                  for the lead, push to
                                                  Instantly, mark "contacted"
                                                  on success / "error" on fail

Both are also called from the legacy ``outreach_agent(state)`` LangGraph
node, which keeps the old fully-automatic ``run_pipeline`` working for
the /run/{lead_id} synchronous endpoint and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from tools import campaigns as campaigns_tool
from tools import events, llm
from tools.email_validator import validate_email
from tools.instantly import send_email
from tools.supabase_client import (
    get_latest_outreach_for_lead,
    insert_outreach,
    log_action,
    update_lead,
    update_outreach_by_lead,
)
from tools.zerobounce import verify_email as zb_verify_email

MODEL = "gpt-4o"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "outreach.txt"

EMAIL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["subject", "body"],
    "properties": {
        "subject": {"type": "string", "minLength": 1, "maxLength": 80},
        "body": {"type": "string", "minLength": 1},
    },
}


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Scribe ✍️ — generate the email but don't send
# --------------------------------------------------------------------------- #


def draft_email_for_lead(
    lead: dict,
    *,
    run_id: str = "",
    lead_index: int = 0,
) -> dict:
    """Generate a cold-email draft for ``lead`` and persist it as 'pending'.

    Returns ``{success, subject, body, error}``.
    """
    lead_id = lead.get("id", "")
    name = lead.get("name", "") or ""
    company = lead.get("company", "") or ""
    industry = lead.get("industry", "") or ""
    website = lead.get("website", "") or ""

    print(f"[SCRIBE]   Drafting email for {company}…")
    events.emit(
        run_id,
        "scribe_start",
        {"model": MODEL, "lead_id": lead_id, "company": company, "index": lead_index},
    )

    # B11 — fold the campaign's ICP context + pitch angle into the prompt
    # so different campaigns produce on-brand emails. When the lead has no
    # campaign attached, inject empty strings (the prompt template is
    # tolerant of those).
    campaign = None
    cid = (lead.get("campaign_id") or "").strip()
    if cid:
        campaign = campaigns_tool.get_campaign(cid)
    campaign_context = ""
    pitch_angle = ""
    if campaign:
        if campaign.get("description"):
            campaign_context = f"Campaign brief / ICP: {campaign['description']}"
        if campaign.get("pitch_angle"):
            pitch_angle = f"Specific pitch angle for this campaign: {campaign['pitch_angle']}"

    prompt = llm.render(
        _load_prompt(),
        name=name,
        company=company,
        industry=industry,
        website=website,
        campaign_context=campaign_context,
        pitch_angle=pitch_angle,
    )

    try:
        result = llm.call_json(
            model=MODEL,
            system="You are a cold email copywriter. Follow the rules in the prompt.",
            user=prompt,
            schema=EMAIL_SCHEMA,
            schema_name="ColdEmail",
            temperature=0.7,
            agent="scribe",
            lead_id=lead_id,
            action="draft",
        )
    except Exception as exc:
        err = f"draft error: {exc}"
        print(f"[SCRIBE]   {err}")
        events.emit(run_id, "scribe_error", {"error": err, "lead_id": lead_id, "index": lead_index})
        return {"success": False, "subject": "", "body": "", "error": err}

    data = result.data or {}
    subject = str(data.get("subject", "")).strip()
    body = str(data.get("body", "")).strip()
    if not subject or not body:
        err = "empty subject or body from model"
        events.emit(run_id, "scribe_error", {"error": err, "lead_id": lead_id, "index": lead_index})
        return {"success": False, "subject": subject, "body": body, "error": err}

    # Persist as pending; Postie will pick this up when the user clicks Send.
    insert_outreach(lead_id, subject, body, status="pending")
    if lead_id:
        update_lead(lead_id, {"status": "drafted"})

    print(f"[SCRIBE]   Subject: {subject}")
    events.emit(
        run_id,
        "scribe_done",
        {
            "lead_id": lead_id,
            "subject": subject,
            "body": body,
            "index": lead_index,
        },
    )
    return {"success": True, "subject": subject, "body": body, "error": ""}


# --------------------------------------------------------------------------- #
# Postie 📮 — fetch the existing draft and send it
# --------------------------------------------------------------------------- #


def send_existing_draft(
    lead: dict,
    *,
    run_id: str = "",
    lead_index: int = 0,
) -> dict:
    """Send the most recent pending/edited draft for ``lead`` via Instantly.

    Returns ``{success, mode, subject, body, error}``.
    """
    lead_id = lead.get("id", "")
    email = lead.get("email", "") or ""

    if not email:
        return {"success": False, "mode": "", "subject": "", "body": "",
                "error": "lead has no email address"}

    out_row = get_latest_outreach_for_lead(lead_id)
    if not out_row:
        return {"success": False, "mode": "", "subject": "", "body": "",
                "error": "no draft found — Scribe hasn't written this yet"}

    subject = out_row.get("email_subject") or ""
    body = out_row.get("email_body") or ""
    if not subject or not body:
        return {"success": False, "mode": "", "subject": subject, "body": body,
                "error": "draft is empty"}

    # Tier-A1: MX gate. If the domain has no mail exchanger, the message will
    # 100% bounce — and bounces destroy sender reputation. Refuse before send.
    print(f"[POSTIE]   Validating {email}…")
    events.emit(
        run_id,
        "postie_validating",
        {"to": email, "lead_id": lead_id, "index": lead_index},
    )
    validation = validate_email(email)
    if not validation.deliverable:
        err = f"email rejected ({validation.verdict}): {validation.error}"
        print(f"[POSTIE]   ✗ {err}")
        update_outreach_by_lead(lead_id, {"status": "failed"})
        if lead_id:
            update_lead(lead_id, {"status": "error"})
        log_action("postie", lead_id, "validate_email", "", err,
                   model="email_validator")
        events.emit(
            run_id,
            "postie_failed",
            {
                "error": err,
                "verdict": validation.verdict,
                "lead_id": lead_id,
                "index": lead_index,
            },
        )
        return {
            "success": False,
            "mode": "validation",
            "subject": subject,
            "body": body,
            "error": err,
        }

    # ---- B10: ZeroBounce mailbox verification ----
    # MX passed → domain is real. Now check the actual mailbox status. ZB
    # rejects spamtraps, abuse addresses, do_not_mail, and confirmed-invalid
    # mailboxes. We "warn" through catch-all/unknown — those are sendable but
    # have higher bounce risk; user already approved the draft, so we proceed.
    print(f"[POSTIE]   ZB-verifying {email}…")
    events.emit(
        run_id,
        "postie_zb_verifying",
        {"to": email, "lead_id": lead_id, "index": lead_index},
    )
    zb = zb_verify_email(email)
    if zb.rejects:
        err = f"ZeroBounce rejected ({zb.status}/{zb.sub_status or '-'}): {zb.error or 'do not send'}"
        print(f"[POSTIE]   ✗ {err}")
        update_outreach_by_lead(lead_id, {"status": "failed"})
        if lead_id:
            update_lead(lead_id, {"status": "error"})
        log_action("postie", lead_id, "zerobounce_verify", "", err,
                   model="zerobounce")
        events.emit(
            run_id,
            "postie_failed",
            {
                "error": err,
                "verdict": "zb_reject",
                "zb_status": zb.status,
                "zb_sub_status": zb.sub_status,
                "lead_id": lead_id,
                "index": lead_index,
            },
        )
        return {
            "success": False,
            "mode": "zerobounce",
            "subject": subject,
            "body": body,
            "error": err,
        }

    print(
        f"[POSTIE]   Delivering to {email} "
        f"(MX: {validation.mx_host} · ZB: {zb.status}{' [cached]' if zb.cached else ''})…"
    )
    events.emit(
        run_id,
        "postie_start",
        {
            "to": email,
            "subject": subject,
            "lead_id": lead_id,
            "index": lead_index,
            "mx_host": validation.mx_host,
            "zb_status": zb.status,
            "zb_verdict": zb.verdict,
            "zb_cached": zb.cached,
        },
    )

    send_result = send_email(email, subject, body)

    if send_result.get("success"):
        # update_outreach_by_lead auto-stamps sent_at when status flips to 'sent'
        update_outreach_by_lead(lead_id, {"status": "sent"})
        if lead_id:
            update_lead(lead_id, {"status": "contacted"})
        log_action(
            "postie",
            lead_id,
            "send_email",
            result=f"mode={send_result.get('mode')} subject={subject}",
        )
        print("[POSTIE]   Delivered ✓")
        events.emit(
            run_id,
            "postie_done",
            {
                "lead_id": lead_id,
                "mode": send_result.get("mode"),
                "to": email,
                "subject": subject,
                "index": lead_index,
            },
        )
        return {
            "success": True,
            "mode": send_result.get("mode", ""),
            "subject": subject,
            "body": body,
            "error": "",
        }

    err = send_result.get("error", "unknown send error")
    update_outreach_by_lead(lead_id, {"status": "failed"})
    if lead_id:
        update_lead(lead_id, {"status": "error"})
    log_action("postie", lead_id, "send_email", "", err)
    print(f"[POSTIE]   Failed ({err})")
    events.emit(
        run_id,
        "postie_failed",
        {"error": err, "lead_id": lead_id, "index": lead_index},
    )
    return {
        "success": False,
        "mode": send_result.get("mode", ""),
        "subject": subject,
        "body": body,
        "error": err,
    }


# --------------------------------------------------------------------------- #
# Legacy LangGraph node — keeps run_pipeline + tests working
# --------------------------------------------------------------------------- #


def outreach_agent(state: dict) -> dict:
    """Auto-flow: draft + send in one shot. Used by the legacy synchronous
    /run/{lead_id} endpoint and the existing test suite."""
    lead = {
        "id": state.get("lead_id", ""),
        "name": state.get("name", ""),
        "email": state.get("email", ""),
        "company": state.get("company", ""),
        "website": state.get("website", ""),
        "industry": state.get("industry", ""),
    }
    run_id = state.get("run_id", "")
    index = state.get("lead_index", 0)

    draft = draft_email_for_lead(lead, run_id=run_id, lead_index=index)
    if not draft["success"]:
        new_state = dict(state)
        new_state["error"] = draft["error"]
        new_state["status"] = "error"
        return new_state

    send = send_existing_draft(lead, run_id=run_id, lead_index=index)
    new_state = dict(state)
    new_state["email_subject"] = draft["subject"]
    new_state["email_body"] = draft["body"]
    if send["success"]:
        new_state["outreach_sent"] = True
        new_state["status"] = "contacted"
    else:
        new_state["outreach_sent"] = False
        new_state["error"] = send["error"]
        new_state["status"] = "error"
    return new_state
