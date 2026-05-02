"""Lead finder agent.

Source priority (configurable via LEAD_FINDER_SOURCE env var):

    "osm"     → OpenStreetMap only (free, real businesses, no key)
    "tavily"  → Tavily web search + GPT extraction (broad, free 1k/mo)
    "apify"   → Instagram + Facebook scraping via Apify (paid, deep)
    "social"  → Tavily + Apify combined → best for social-first regions
                (West Bank, Gaza, MENA, anywhere Google Maps is thin)
    "openai"  → OpenAI generation only (made-up — DEMO ONLY)
    "auto"    → OSM → Tavily → Apify → OpenAI, in order, until something
                returns leads (default).

Lead shape returned to the rest of the pipeline:
    {name, company, email, website, instagram, facebook, phone, industry}

Leads without an email cannot be reached by Instantly — they are still
returned and the pipeline tags them ``status=no_contact_email`` so they
appear in the dashboard for manual DM outreach.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from tools import apify_social, events, llm, osm, tavily

load_dotenv(override=True)

MODEL = "gpt-4o-mini"
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "finder.txt"
SOURCE = os.getenv("LEAD_FINDER_SOURCE", "auto").strip().lower()

# Schema for the OpenAI fallback generator. Loose on optional channels but
# strict on the wrapper shape so json.loads is always safe.
FINDER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["leads"],
    "properties": {
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "company", "email", "website", "industry"],
                "properties": {
                    "name":     {"type": "string"},
                    "company":  {"type": "string"},
                    "email":    {"type": "string"},
                    "website":  {"type": "string"},
                    "industry": {"type": "string"},
                },
            },
        },
    },
}


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _normalise(item: dict, niche: str) -> dict:
    """Make sure every lead carries the full set of fields (empty strings ok).

    Schema fields are explicitly listed; any "_meta" / "latest_post_iso" /
    "posts_count" etc. that a source attaches passes through untouched so
    the enricher can use them downstream (these aren't persisted to DB).
    """
    schema = {
        "name": str(item.get("name", "")).strip() or "Owner",
        "company": str(item.get("company", "")).strip() or "Unknown Co",
        "email": str(item.get("email", "")).strip(),
        "website": str(item.get("website", "")).strip(),
        "instagram": str(item.get("instagram", "")).strip().lstrip("@"),
        "facebook": str(item.get("facebook", "")).strip(),
        "phone": str(item.get("phone", "")).strip(),
        "industry": str(item.get("industry", "")).strip() or niche,
        "source": str(item.get("source", "")).strip(),
    }
    # Pass-through transient meta (consumed by enricher, never persisted).
    for k in ("latest_post_iso", "posts_count", "followers_count"):
        if k in item and item[k] not in (None, ""):
            schema[k] = item[k]
    return schema


def _dedupe(leads: list[dict], cap: int) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for l in leads:
        key = (l.get("company") or "").lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(l)
        if len(out) >= cap:
            break
    return out


def _generate_via_openai(location: str, niche: str, count: int) -> list[dict]:
    prompt = llm.render(
        _load_prompt(), location=location, niche=niche, count=str(count)
    )
    result = llm.call_json(
        model=MODEL,
        system="You generate plausible-looking but fictional businesses for demo data.",
        user=prompt,
        schema=FINDER_SCHEMA,
        schema_name="LeadList",
        temperature=0.9,
        agent="lead_finder",
        action="generate_openai",
    )
    leads = (result.data or {}).get("leads", [])
    if not isinstance(leads, list):
        return []
    return [item for item in leads[:count] if isinstance(item, dict)]


def _try(source_name: str, fn, run_id: str, location: str, niche: str, count: int) -> list[dict]:
    """Run a source function with consistent logging + event emission.

    Every lead returned is tagged with ``source=<source_name>`` so the rest
    of the pipeline can track which source produced what (Tier-A1 provenance).
    """
    print(f"[FINDER]   -> {source_name}")
    events.emit(run_id, "finder_progress", {"step": f"{source_name}_query"})
    try:
        leads = fn(location, niche, count) or []
    except Exception as exc:
        print(f"[FINDER]   {source_name} error: {exc}")
        events.emit(
            run_id,
            "finder_progress",
            {"step": f"{source_name}_error", "error": str(exc)},
        )
        return []
    out: list[dict] = []
    for l in leads:
        if not isinstance(l, dict):
            continue
        norm = _normalise(l, niche)
        norm["source"] = source_name
        out.append(norm)
    if out:
        print(f"[FINDER]   {source_name} returned {len(out)} leads")
    return out


def find_leads(
    location: str,
    niche: str,
    count: int = 5,
    run_id: str = "",
) -> list[dict]:
    """Return real leads from the configured source(s)."""
    count = max(1, min(int(count or 5), 20))
    events.emit(
        run_id,
        "finder_start",
        {
            "location": location,
            "niche": niche,
            "count": count,
            "source": SOURCE,
        },
    )
    print(f"[FINDER]   Source={SOURCE} · searching {count} '{niche}' in {location}")

    leads: list[dict] = []
    used_sources: list[str] = []

    def add(name: str, fn) -> None:
        nonlocal leads
        if len(leads) >= count:
            return
        remaining = count - len(leads)
        new = _try(name, fn, run_id, location, niche, remaining)
        if new:
            used_sources.append(name)
            leads = _dedupe(leads + new, count)

    if SOURCE == "osm":
        add("osm", osm.find_businesses)

    elif SOURCE == "tavily":
        add("tavily", tavily.find_businesses)

    elif SOURCE == "apify":
        add("apify", apify_social.find_businesses)

    elif SOURCE == "social":
        # Tavily first (cheap), then Apify to fill the rest.
        add("tavily", tavily.find_businesses)
        add("apify", apify_social.find_businesses)

    elif SOURCE == "openai":
        add("openai", _generate_via_openai)

    else:  # "auto" or unknown
        add("osm", osm.find_businesses)
        add("tavily", tavily.find_businesses)
        add("apify", apify_social.find_businesses)
        if not leads:
            add("openai", _generate_via_openai)

    if not leads:
        events.emit(run_id, "finder_error", {"error": "no leads found from any source"})
        print("[FINDER]   no leads found")
        return []

    print(f"[FINDER]   Found {len(leads)} leads (sources={','.join(used_sources)})")
    events.emit(
        run_id,
        "finder_complete",
        {
            "count": len(leads),
            "leads": leads,
            "source": ",".join(used_sources) or SOURCE,
        },
    )
    return leads
