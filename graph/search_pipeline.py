"""Multi-lead search pipeline (human-in-the-loop version).

Phase ordering (B7 — parallel enrichment):
    Phase A — sequential, fast: dedupe, insert, no-email shortcut.
    Phase B — parallel:           enrich every remaining lead concurrently
                                  (HTTP-bound, fans out via ThreadPoolExecutor).
    Phase C — sequential, slow:   backfill + score per lead (LLM-bound).

Why this split:
    Enrichment is HTTP — N concurrent fetches finish in roughly the time of
    the slowest one. Scoring calls OpenAI sequentially because each call is
    independently rate-limited and we want predictable token-spend ordering.

It then **stops**. Drafting and sending are manual — the user hits ✍️ "Draft"
and 📮 "Send" buttons, which call ``/api/draft/{lead_id}`` and
``/api/send/{lead_id}``.

Lead status after this pipeline:
    qualified         — score >= 3, awaiting Scribe
    rejected          — score <  3 (or kill-switch: closed / parked)
    no_contact_email  — no email available, queue for manual DM
    error             — scorer failed
"""

from __future__ import annotations

from typing import Optional

from agents.enricher import enrich_leads_parallel
from agents.lead_finder import find_leads
from agents.lead_scorer import lead_scorer_agent
from tools import campaigns as campaigns_tool
from tools import events
from tools.supabase_client import (
    _domain_of,
    insert_lead,
    lead_exists_for_domain,
    update_lead,
)

DEDUPE_TTL_DAYS = 90


def run_search(
    location: str,
    niche: str,
    count: int,
    run_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
) -> dict:
    """Find leads + score each. Returns summary; emails are NOT drafted/sent.

    ``campaign_id`` (B11): when set, the lead is tagged with the campaign and
    the scorer reads campaign-specific weights + threshold from the row.
    """
    # Resolve campaign once (avoids a per-lead lookup) and pass to scorer state.
    campaign = campaigns_tool.get_campaign(campaign_id) if campaign_id else None

    summary = {
        "location": location,
        "niche": niche,
        "campaign_id": campaign_id,
        "campaign_name": (campaign or {}).get("name", ""),
        "found": 0,
        "qualified": 0,
        "rejected": 0,
        "needs_dm": 0,
        "duplicates": 0,
        "errors": 0,
    }

    # Wrap the entire body so a crash anywhere still emits search_done.
    # Without this, an unhandled exception leaves cards stuck at
    # "awaiting enricher…" because the SSE stream closes without a
    # terminal event.
    try:
        return _run_search_inner(location, niche, count, run_id, campaign, summary)
    except Exception as exc:
        print(f"[SEARCH]   FATAL: {exc}")
        summary["errors"] += 1
        events.emit(run_id, "search_error", {"error": str(exc)})
        events.emit(run_id, "search_done", {"summary": summary})
        return summary


def _run_search_inner(
    location: str,
    niche: str,
    count: int,
    run_id: Optional[str],
    campaign: Optional[dict],
    summary: dict,
) -> dict:
    campaign_id = (campaign or {}).get("id")
    leads = find_leads(location, niche, count, run_id=run_id)
    summary["found"] = len(leads)
    if not leads:
        events.emit(run_id, "search_done", {"summary": summary})
        return summary

    # ---- Phase A — dedupe, insert, no-email shortcut ----
    # Each entry in ``processable`` is the tuple of bits we'll need in
    # phase C (after enrichment): (index, merged_lead_for_enricher, stored).
    processable: list[tuple[int, dict, dict]] = []

    for index, lead in enumerate(leads):
        # A1: 90-day domain dedupe before any tokens spent.
        domain = _domain_of(lead.get("website", "") or lead.get("email", ""))
        if domain:
            existing = lead_exists_for_domain(domain, within_days=DEDUPE_TTL_DAYS)
            if existing:
                summary["duplicates"] += 1
                events.emit(
                    run_id,
                    "lead_skipped",
                    {
                        "index": index,
                        "lead": lead,
                        "reason": "duplicate",
                        "domain": domain,
                        "existing_lead_id": existing.get("id"),
                        "existing_status": existing.get("status"),
                    },
                )
                print(
                    f"[FINDER]   ↷ skip {lead.get('company','?')} "
                    f"(dupe of {domain}, status={existing.get('status')})"
                )
                continue

        # B11: tag the lead with the campaign so reporting + the dashboard
        # can filter by campaign later.
        if campaign_id:
            lead = {**lead, "campaign_id": campaign_id}

        try:
            stored = insert_lead(lead)
        except Exception as exc:
            events.emit(
                run_id,
                "lead_skipped",
                {"index": index, "lead": lead, "error": str(exc)},
            )
            summary["errors"] += 1
            continue

        events.emit(run_id, "lead_start", {"index": index, "lead": stored})

        # No email = queue for manual DM, skip the rest.
        if not (stored.get("email") or "").strip():
            update_lead(stored.get("id", ""), {"status": "no_contact_email"})
            summary["needs_dm"] += 1
            events.emit(
                run_id,
                "lead_done",
                {
                    "index": index,
                    "lead_id": stored.get("id", ""),
                    "status": "no_contact_email",
                    "score": None,
                    "instagram": stored.get("instagram", ""),
                    "facebook": stored.get("facebook", ""),
                    "phone": stored.get("phone", ""),
                },
            )
            continue

        # Merge: stored has canonical id + db-shaped fields; original lead
        # carries transient meta (latest_post_iso, posts_count, etc.) the
        # DB schema doesn't store. Stored wins on overlap.
        enriched_input = {**lead, **stored}
        processable.append((index, enriched_input, stored))

    # ---- Phase B — parallel enrichment ----
    enrichments = enrich_leads_parallel(
        [(idx, inp) for idx, inp, _ in processable],
        run_id=run_id,
    )

    # ---- Phase C — sequential backfill + scoring ----
    for index, _enriched_input, stored in processable:
        enrichment = enrichments.get(index, {})

        # Backfill any contact channels the enricher discovered on the page.
        backfill: dict = {}
        if enrichment.get("found_instagram") and not stored.get("instagram"):
            backfill["instagram"] = enrichment["found_instagram"]
        if enrichment.get("found_facebook") and not stored.get("facebook"):
            backfill["facebook"] = enrichment["found_facebook"]
        if enrichment.get("found_phone") and not stored.get("phone"):
            backfill["phone"] = enrichment["found_phone"]
        if backfill and stored.get("id"):
            update_lead(stored["id"], backfill)
            stored.update(backfill)

        scorer_state = {
            "lead_id": stored.get("id", ""),
            "run_id": run_id or "",
            "lead_index": index,
            "name": stored.get("name", "") or "",
            "email": stored.get("email", "") or "",
            "company": stored.get("company", "") or "",
            "website": stored.get("website", "") or "",
            "industry": stored.get("industry", "") or "",
            "instagram": stored.get("instagram", "") or "",
            "facebook": stored.get("facebook", "") or "",
            "phone": stored.get("phone", "") or "",
            "signals": enrichment.get("signals", {}),
            "enrichment": enrichment,
            # B11: scorer reads campaign-specific rules from state
            "campaign": campaign,
        }
        out = lead_scorer_agent(scorer_state)
        status = out.get("status")
        score = out.get("score", 0)
        if out.get("error"):
            summary["errors"] += 1
        elif status == "qualified":
            summary["qualified"] += 1
        elif status == "rejected":
            summary["rejected"] += 1

        events.emit(
            run_id,
            "lead_done",
            {
                "index": index,
                "lead_id": stored.get("id", ""),
                "status": status,
                "score": score,
                "reasoning": out.get("reasoning", ""),
            },
        )

    events.emit(run_id, "search_done", {"summary": summary})
    return summary
