"""Lead scorer agent — deterministic rules first, LLM for reasoning only.

Tier-2 redesign: scoring is now grounded in real evidence from the enricher,
not LLM intuition over a company name.

Pipeline:
    1. Deterministic rule layer maps signals -> base_score (0..5)
    2. LLM gets the signals as evidence and writes a one-sentence
       reasoning that cites them. The LLM does NOT change the score.

Why this split?
    Pure-LLM scoring is noisy: same lead can score 3 vs 4 on consecutive
    runs, and the reasoning is unverifiable. With deterministic scoring
    the same evidence always produces the same score, and the LLM is
    constrained to narrating what the rules already decided.

If the enricher couldn't run (e.g. lead has no website at all and the
caller skipped enrichment), the scorer falls back to a minimal score
based on what's in the lead dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools import events, llm
from tools.supabase_client import update_lead

MODEL = "gpt-4o-mini"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "scorer.txt"

REASONING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reasoning"],
    "properties": {
        "reasoning": {"type": "string", "minLength": 1, "maxLength": 280},
    },
}


# ---- Deterministic rule layer -------------------------------------------- #

# Each signal maps to a (score_delta, human_label) tuple. Positive deltas
# push the score UP (lead more likely to need us), negative push DOWN.
# Numbers were chosen so a "no website" lead floors at ~3 (qualified) and a
# "modern site, fast, https, mobile" lead floors at 0-1 (rejected).
SIGNAL_WEIGHTS: dict[str, tuple[int, str]] = {
    "no_website":         (+3, "No website at all — huge gap"),
    "website_dead":       (+3, "Website is broken or unreachable"),
    "outdated_tech":      (+1, "Built on a cheap template (Wix/Squarespace/etc)"),
    "slow_site":          (+1, "Site is slow or heavy"),
    "no_mobile_viewport": (+1, "Not mobile-optimised"),
    "no_https":           (+1, "Missing HTTPS"),
    "active_social":      (+1, "Active on social — engaged audience to convert"),
    "modern_site":        (-2, "Already on a modern stack — less urgent"),
    "has_contact_form":   ( 0, "Has a contact form — neutral, just info"),
    # A2 reliability layer
    "business_stale":     (+1, "Site footer year is years out of date — neglected"),
    "social_silent":      (-1, "Social account hasn't posted in months — possibly dormant"),
}

# A2 — kill switches. Any of these means we should NOT pursue this lead at
# all, regardless of other signals: there's likely no business to email.
# The scorer short-circuits to score=0, status="rejected" before any LLM call.
KILL_SWITCH_SIGNALS: dict[str, str] = {
    "business_closed":    "Business appears permanently closed — explicit closure language on site",
    "parked_domain":      "Domain is parked / for sale — no real business behind it",
}


def _resolve_weights(campaign: Optional[dict]) -> dict[str, tuple[int, str]]:
    """Merge global SIGNAL_WEIGHTS with the campaign's optional override.

    Override format (from campaigns.signal_weights JSON):
        {"slow_site": 2}                  # int → keep label, change delta
        {"slow_site": [2, "Speed kills conversion for ecom"]}  # tuple → both
    """
    if not campaign or not campaign.get("signal_weights"):
        return SIGNAL_WEIGHTS
    merged = dict(SIGNAL_WEIGHTS)
    overrides = campaign.get("signal_weights") or {}
    for key, raw in overrides.items():
        existing_label = merged.get(key, (0, ""))[1]
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            merged[key] = (int(raw[0]), str(raw[1]) or existing_label)
        elif isinstance(raw, (int, float)):
            merged[key] = (int(raw), existing_label)
        # silently ignore malformed entries
    return merged


def _score_from_signals(
    signals: dict[str, bool],
    weights: Optional[dict[str, tuple[int, str]]] = None,
) -> tuple[int, list[str], list[str]]:
    """Return (base_score, positives, negatives) using ``weights`` (or defaults).

    base_score is clamped to [0, 5]. positives/negatives are human-readable
    strings naming each contributing signal so they can land in the prompt
    and be displayed in logs.
    """
    table = weights or SIGNAL_WEIGHTS
    score = 0
    positives: list[str] = []
    negatives: list[str] = []
    for key, present in (signals or {}).items():
        if not present:
            continue
        delta, label = table.get(key, (0, ""))
        if delta > 0:
            score += delta
            positives.append(label)
        elif delta < 0:
            score += delta  # negative
            negatives.append(label)
    return max(0, min(5, score)), positives, negatives


# ---- LLM reasoning layer ------------------------------------------------- #


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _channels(state: dict) -> str:
    bits = []
    if state.get("instagram"): bits.append(f"IG @{state['instagram']}")
    if state.get("facebook"):  bits.append("Facebook")
    if state.get("phone"):     bits.append("phone")
    if state.get("email"):     bits.append("email")
    return ", ".join(bits) or "(none)"


def _bullet(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items) if items else "  (none)"


# ---- Agent --------------------------------------------------------------- #


def lead_scorer_agent(state: dict) -> dict:
    """Score a lead using deterministic rules + LLM reasoning.

    B11: when ``state["campaign"]`` is set, the campaign's signal_weights
    override the global SIGNAL_WEIGHTS for this lead, and the campaign's
    score_threshold replaces the default cutoff of 3.
    """
    lead_id = state.get("lead_id", "")
    run_id = state.get("run_id", "")
    index = state.get("lead_index", 0)
    company = state.get("company", "") or ""

    signals = state.get("signals") or {}
    enrichment = state.get("enrichment") or {}
    campaign = state.get("campaign") or None

    weights = _resolve_weights(campaign)
    threshold = int((campaign or {}).get("score_threshold", 3))
    threshold = max(0, min(5, threshold))

    print("[SCORER]   Computing deterministic score…")
    events.emit(
        run_id,
        "scorer_start",
        {
            "model": MODEL, "company": company, "index": index,
            "campaign": (campaign or {}).get("name", ""),
            "threshold": threshold,
        },
    )

    # ---- A2 kill switch: hard reject if business is closed / domain parked.
    # We skip the LLM entirely (saves tokens) and short-circuit straight to
    # rejected with an evidence-grounded reason.
    for sig_key, reason in KILL_SWITCH_SIGNALS.items():
        if signals.get(sig_key):
            print(f"[SCORER]   ✗ kill-switch ({sig_key}): {reason}")
            events.emit(
                run_id,
                "scorer_complete",
                {
                    "score": 0,
                    "status": "rejected",
                    "reasoning": reason,
                    "signals": signals,
                    "kill_switch": sig_key,
                    "index": index,
                },
            )
            if lead_id:
                update_lead(lead_id, {"score": 0, "status": "rejected"})
            new_state = dict(state)
            new_state["score"] = 0
            new_state["status"] = "rejected"
            new_state["reasoning"] = reason
            new_state["kill_switch"] = sig_key
            return new_state

    # 1. Deterministic layer (campaign-aware)
    score, positives, negatives = _score_from_signals(signals, weights=weights)

    # 2. LLM reasoning layer (purely narrates the evidence — never changes score)
    prompt = llm.render(
        _load_prompt(),
        company=company,
        industry=state.get("industry", "") or "",
        website=state.get("website", "") or "(none)",
        channels=_channels(state),
        base_score=str(score),
        positives=_bullet(positives),
        negatives=_bullet(negatives),
        tech=enrichment.get("tech", "") or "(unknown)",
        load_seconds=str(enrichment.get("load_seconds", 0.0)),
    )

    reasoning = ""
    error = ""
    try:
        result = llm.call_json(
            model=MODEL,
            system="You justify pre-computed lead scores in one short, evidence-grounded sentence.",
            user=prompt,
            schema=REASONING_SCHEMA,
            schema_name="ScoreReasoning",
            temperature=0.3,
            agent="lead_scorer",
            lead_id=lead_id,
            action="reasoning",
        )
        reasoning = str((result.data or {}).get("reasoning", "")).strip()
    except Exception as exc:
        # Reasoning failure is non-fatal — we already have a deterministic score
        error = f"reasoning error: {exc}"
        # Fallback: stitch a sentence from the signals
        if positives:
            reasoning = f"Score {score}/5 — {positives[0].lower()}"
        elif negatives:
            reasoning = f"Score {score}/5 — {negatives[0].lower()}"
        else:
            reasoning = f"Score {score}/5 — no strong signals either way"
        events.emit(run_id, "scorer_warning", {"warning": error, "index": index})

    status = "qualified" if score >= threshold else "rejected"
    print(f"[SCORER]   {score}/5 · {status} · {reasoning}")
    events.emit(
        run_id,
        "scorer_complete",
        {
            "score": score,
            "status": status,
            "reasoning": reasoning,
            "signals": signals,
            "positives": positives,
            "negatives": negatives,
            "index": index,
        },
    )

    if lead_id:
        update_lead(lead_id, {"score": score, "status": status})

    new_state = dict(state)
    new_state["score"] = score
    new_state["status"] = status
    new_state["reasoning"] = reasoning
    if error:
        # Non-fatal: keep the score, just note the warning
        new_state["scorer_warning"] = error
    return new_state
