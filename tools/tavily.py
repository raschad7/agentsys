"""Tavily-powered lead discovery — searches the open web (incl. IG/FB profiles
that surface in search results) and uses GPT to extract structured leads.

Why this fits regions like the West Bank, Gaza, MENA broadly:
    OSM is empty there; many businesses live only on Instagram / Facebook /
    a stand-alone website. Tavily gives clean web search results — including
    social profiles — without dealing with Meta's anti-scraping wall.

Free tier: 1,000 searches / month. Paid is ~$0.008 / search after that.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

from tools import llm

load_dotenv(override=True)

TAVILY_URL = "https://api.tavily.com/search"
EXTRACT_MODEL = "gpt-4o-mini"

# JSON schema for the extractor — guarantees the shape we map into the
# pipeline's lead dict, no defensive coding needed downstream.
EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["leads"],
    "properties": {
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["company", "website", "instagram", "facebook",
                             "email", "phone", "industry"],
                "properties": {
                    "company":   {"type": "string"},
                    "website":   {"type": "string"},
                    "instagram": {"type": "string"},
                    "facebook":  {"type": "string"},
                    "email":     {"type": "string"},
                    "phone":     {"type": "string"},
                    "industry":  {"type": "string"},
                },
            },
        },
    },
}


def _key() -> str:
    return os.getenv("TAVILY_API_KEY", "").strip()


def _build_queries(location: str, niche: str) -> list[str]:
    """Variations chosen to surface different parts of the web.

    The first hits Instagram/Facebook profiles directly, the second targets
    business websites, the third grabs general listings + directories.
    """
    return [
        f'"{niche}" {location} (site:instagram.com OR site:facebook.com)',
        f'"{niche}" {location} contact email',
        f"best {niche} in {location}",
    ]


def _tavily_search(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    api_key = _key()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(TAVILY_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data.get("results", []) or []


_EXTRACT_SYSTEM = (
    "You extract real businesses from web search results. Output valid JSON only. "
    "Never invent businesses — only return what is clearly present in the snippets."
)


def _extract_prompt(location: str, niche: str, results: list[dict], wanted: int) -> str:
    blob = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('content','')[:400]}"
        for i, r in enumerate(results)
    )
    return (
        f"You are looking for real {niche} businesses in {location}.\n"
        f"Below are real web search results. Extract up to {wanted} distinct businesses.\n\n"
        f"For each business, return:\n"
        f"  - company:   business name\n"
        f"  - website:   website URL if any (not an instagram/facebook URL)\n"
        f"  - instagram: instagram handle WITHOUT @ (e.g. 'cafe_x') if visible\n"
        f"  - facebook:  facebook page URL if visible\n"
        f"  - email:     email if visible in the snippet, else empty\n"
        f"  - phone:     phone if visible, else empty\n"
        f"  - industry:  '{niche}'\n\n"
        f"Skip:\n"
        f"  - Generic 'top 10' listicles that are not a single business\n"
        f"  - Yelp/TripAdvisor/Google Maps result pages (these list many)\n"
        f"  - Anything you're not confident is a real, single business\n\n"
        f"Output JSON: {{\"leads\": [ ... ]}}\n\n"
        f"SEARCH RESULTS:\n{blob}"
    )


def _gpt_extract(location: str, niche: str, results: list[dict], wanted: int) -> list[dict]:
    if not results:
        return []
    try:
        result = llm.call_json(
            model=EXTRACT_MODEL,
            system=_EXTRACT_SYSTEM,
            user=_extract_prompt(location, niche, results, wanted),
            schema=EXTRACT_SCHEMA,
            schema_name="ExtractedLeads",
            temperature=0.2,
            agent="tavily",
            action="extract",
        )
    except Exception as exc:
        print(f"[tavily] extraction failed: {exc}")
        return []

    leads = (result.data or {}).get("leads", [])
    if not isinstance(leads, list):
        return []
    out: list[dict] = []
    for item in leads:
        if not isinstance(item, dict):
            continue
        company = str(item.get("company", "")).strip()
        if not company:
            continue
        out.append(
            {
                "name": "Owner",  # web search rarely surfaces owner names
                "company": company,
                "email": str(item.get("email", "")).strip(),
                "website": str(item.get("website", "")).strip(),
                "instagram": str(item.get("instagram", "")).strip().lstrip("@"),
                "facebook": str(item.get("facebook", "")).strip(),
                "phone": str(item.get("phone", "")).strip(),
                "industry": str(item.get("industry", "")).strip() or niche,
            }
        )
    return out


def find_businesses(location: str, niche: str, count: int = 5) -> list[dict]:
    """Search Tavily across multiple query angles, then GPT-extract leads.

    Returns a deduped list (by company name) capped at ``count``.
    """
    if not _key():
        print("[tavily] TAVILY_API_KEY missing — skipping")
        return []

    all_results: list[dict] = []
    seen_urls: set[str] = set()
    for q in _build_queries(location, niche):
        try:
            hits = _tavily_search(q, max_results=8)
        except Exception as exc:
            print(f"[tavily] search failed for {q!r}: {exc}")
            continue
        for h in hits:
            url = h.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(h)
        if len(all_results) >= count * 4:
            break

    if not all_results:
        return []

    print(f"[tavily]   {len(all_results)} raw results → extracting with GPT")
    try:
        leads = _gpt_extract(location, niche, all_results, count * 2)
    except Exception as exc:
        print(f"[tavily] extraction failed: {exc}")
        return []

    # Dedupe by lowercased company name
    seen: set[str] = set()
    unique: list[dict] = []
    for lead in leads:
        key = lead["company"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(lead)
        if len(unique) >= count:
            break
    return unique
