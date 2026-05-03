"""DM agent — drafts WhatsApp / Instagram / Facebook conversation messages.

Different from Scribe (cold email): DMs are SHORT, casual, single-paragraph.
They open with a question that's easy to reply to. They never include a
formal sign-off. Two-line max for the opener.

Two modes:
    draft_opener(lead, channel, campaign?) -> str
        First message — soft, curious, references something specific
        about their business so it doesn't read like spam.

    draft_reply(lead, channel, history, last_them_message) -> str
        Continues the conversation. Adapts tone to theirs (if they
        replied in Arabic, we reply in Arabic; if formal, we go formal).
        Goal is always: get the call booked.
"""

from __future__ import annotations

from typing import Optional

from tools import campaigns as campaigns_tool
from tools import llm

MODEL = "gpt-4o-mini"

# JSON schemas for structured outputs — guarantees we always get one
# 'message' string back, no preambles or chatter.
_OPENER_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}
_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "intent": {
            "type": "string",
            "enum": ["continue", "book_call", "qualify", "price", "objection", "wrap_up"],
        },
    },
    "required": ["message", "intent"],
    "additionalProperties": False,
}


def _channel_voice(channel: str) -> str:
    """Tone hints per channel — IG is more casual, WA more direct."""
    return {
        "whatsapp": "WhatsApp DM. Direct, friendly, one or two short sentences max. No emojis unless the lead used them first. No subject line.",
        "instagram": "Instagram DM. Casual, like a real person sliding into DMs. Reference something visible on their profile (followers count, bio, niche) when natural.",
        "facebook": "Facebook page message. Slightly more formal than IG but still brief. One short paragraph.",
    }.get(channel, "Short, friendly DM. One paragraph.")


def _campaign_blurb(lead: dict) -> str:
    cid = (lead.get("campaign_id") or "").strip()
    if not cid:
        return ""
    c = campaigns_tool.get_campaign(cid) or {}
    parts = []
    if c.get("description"):
        parts.append(f"Campaign brief / ICP: {c['description']}")
    if c.get("pitch_angle"):
        parts.append(f"Pitch angle: {c['pitch_angle']}")
    return "\n".join(parts)


def draft_opener(lead: dict, channel: str) -> dict:
    """Draft the very first DM. Returns ``{message, error}``."""
    company = lead.get("company") or "your business"
    industry = lead.get("industry") or ""
    instagram = lead.get("instagram") or ""
    website = lead.get("website") or ""
    bio_hint = ""
    # Apify enricher leaves a `_bio` scratch field when running through
    # find_businesses, but it's stripped before insert; if present from a
    # fresh search we use it for color.
    if lead.get("_bio"):
        bio_hint = f"\nIG bio snippet: {lead['_bio']}"

    voice = _channel_voice(channel)
    campaign_blurb = _campaign_blurb(lead)

    system = (
        "You are a friendly outbound rep for a small web-design agency. "
        "You write short DMs that open the door for a real conversation. "
        "You sound like a person, not a marketing template. "
        "NEVER use phrases like 'I came across', 'I noticed', 'Just wanted to reach out'. "
        "Open with a specific observation or a soft question, not a pitch."
    )

    user = (
        f"Voice: {voice}\n\n"
        f"Lead:\n"
        f"  Company: {company}\n"
        f"  Industry: {industry}\n"
        f"  Instagram: {instagram or '-'}\n"
        f"  Website: {website or '(none)'}\n"
        f"{bio_hint}\n\n"
        f"{campaign_blurb}\n\n"
        f"Channel: {channel}\n\n"
        "Write the FIRST DM to send to this business. "
        "Two-line max. Friendly but direct. "
        "If they have no website, gently hint a website might help — but "
        "don't make it a pitch yet, just a curious question. "
        "If their IG/site exists, reference one specific thing about it. "
        "Output JSON: { \"message\": <your DM text> }"
    )

    try:
        result = llm.call_json(
            model=MODEL,
            system=system,
            user=user,
            schema=_OPENER_SCHEMA,
            schema_name="DMOpener",
            temperature=0.8,
            agent="dm_agent",
            lead_id=lead.get("id", ""),
            action="draft_opener",
        )
    except Exception as exc:
        return {"message": "", "error": f"opener draft failed: {exc}"}

    msg = (result.data or {}).get("message", "").strip()
    if not msg:
        return {"message": "", "error": "model returned empty message"}
    return {"message": msg, "error": ""}


def draft_reply(
    lead: dict,
    channel: str,
    turns: list[dict],
    last_them: Optional[str] = None,
) -> dict:
    """Draft the next outgoing message based on conversation history.

    ``turns`` is the list from ``conversations.turns`` — each item has
    ``{role, message, ts}``. ``last_them`` is the most recent message
    from them (already in turns, but pulled out for emphasis).

    Returns ``{message, intent, error}``.
    """
    company = lead.get("company") or "the business"
    voice = _channel_voice(channel)
    campaign_blurb = _campaign_blurb(lead)

    # Render history compactly so the model sees the whole arc.
    history_lines = []
    for t in turns[-12:]:  # cap context at last 12 turns
        speaker = "Us" if t.get("role") == "us" else "Them"
        history_lines.append(f"{speaker}: {t.get('message','').strip()}")
    history = "\n".join(history_lines) or "(no history yet)"

    last_them_block = ""
    if last_them:
        last_them_block = f"\nTheir most recent message:\n  {last_them}\n"

    system = (
        "You are continuing a real DM conversation between a web-design agency rep "
        "and a small business owner. Match the LAST message's language and tone "
        "exactly — if they replied in Arabic, reply in Arabic; if casual, stay "
        "casual; if formal, match formal. Your goal is to move toward booking a "
        "short call (10-15 min) where you can show example work. Don't pitch "
        "in the DM — keep the DM short and book the call."
    )

    user = (
        f"Voice: {voice}\n"
        f"Lead: {company} (industry: {lead.get('industry') or '-'})\n"
        f"{campaign_blurb}\n\n"
        f"Conversation so far:\n{history}\n"
        f"{last_them_block}\n"
        "Write the next message FROM US. Rules:\n"
        "  - 1-3 short sentences max\n"
        "  - mirror their language (English/Arabic/etc.) and tone\n"
        "  - if they showed interest, propose a specific time slot or ask 'when works for you this week?'\n"
        "  - if they raised an objection, acknowledge it briefly then redirect\n"
        "  - if they asked about price, give a soft range and offer to discuss on a call\n"
        "  - never use marketing fluff or formal sign-offs\n\n"
        "Also tag the message with one intent:\n"
        "  continue   — keeping the convo alive\n"
        "  book_call  — proposing/confirming a call\n"
        "  qualify    — asking a discovery question\n"
        "  price      — handling a price question\n"
        "  objection  — handling an objection\n"
        "  wrap_up    — politely closing if they're not interested\n\n"
        "Output JSON: { \"message\": <text>, \"intent\": <one of above> }"
    )

    try:
        result = llm.call_json(
            model=MODEL,
            system=system,
            user=user,
            schema=_REPLY_SCHEMA,
            schema_name="DMReply",
            temperature=0.7,
            agent="dm_agent",
            lead_id=lead.get("id", ""),
            action="draft_reply",
        )
    except Exception as exc:
        return {"message": "", "intent": "", "error": f"reply draft failed: {exc}"}

    data = result.data or {}
    msg = (data.get("message") or "").strip()
    intent = (data.get("intent") or "continue").strip()
    if not msg:
        return {"message": "", "intent": "", "error": "model returned empty message"}
    return {"message": msg, "intent": intent, "error": ""}
