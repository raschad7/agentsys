"""Apify-powered social scraping (Instagram + Facebook).

We call two ready-made Apify actors via their sync ``run-sync-get-dataset-items``
endpoint, which blocks until the actor finishes and returns the dataset items
in one HTTP call — perfect for our agent flow.

  Instagram:  apify/instagram-search-scraper
              → search by keyword/hashtag, returns profiles with bio,
                public email (if exposed via Contact button), website link,
                follower count.

  Facebook:   apify/facebook-pages-scraper
              → search pages by query, returns email, phone, website, likes.

Cost guardrails (important — Apify charges per result):
  - Hard cap via ``APIFY_MAX_RESULTS`` env var (default 20).
  - We split the cap evenly between IG + FB.
  - On any error we return [], we never retry (so you can't accidentally
    burn through your credit by clicking "Search" repeatedly on a broken setup).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

APIFY_BASE = "https://api.apify.com/v2"
IG_ACTOR = "apify~instagram-search-scraper"
FB_ACTOR = "apify~facebook-pages-scraper"


def _token() -> str:
    return os.getenv("APIFY_TOKEN", "").strip()


def _max_results() -> int:
    raw = os.getenv("APIFY_MAX_RESULTS", "20").strip()
    try:
        return max(1, min(int(raw), 100))
    except ValueError:
        return 20


def _run_actor_sync(actor_id: str, payload: dict, timeout: float = 120.0) -> list[dict]:
    """Run an Apify actor synchronously and return its dataset items.

    On non-2xx responses, capture the response body in the exception so the
    caller (and the operator reading the logs) can see *why* Apify said no.
    httpx's default error message hides the body, which is useless for
    debugging actor input shape mismatches.
    """
    token = _token()
    if not token:
        raise RuntimeError("APIFY_TOKEN not set")
    url = f"{APIFY_BASE}/acts/{actor_id}/run-sync-get-dataset-items"
    params = {"token": token}
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, params=params, json=payload)
        if resp.status_code >= 400:
            body = (resp.text or "")[:600]
            raise RuntimeError(
                f"Apify {actor_id} → HTTP {resp.status_code}: {body}"
            )
        data = resp.json()
    return data if isinstance(data, list) else []


# ---- Instagram ------------------------------------------------------------ #


def _scrape_instagram(location: str, niche: str, limit: int) -> list[dict]:
    """Search Instagram for profiles matching the niche + location.

    The `apify/instagram-search-scraper` input schema wants:
        search: string (not array)
        searchType: "user" | "hashtag" | "place"
        searchLimit: int
    Sending search as a list returns HTTP 400 — schema validation reject.
    """
    if limit <= 0:
        return []
    payload = {
        "search": f"{niche} {location}",
        "searchType": "user",   # profiles, not posts
        "searchLimit": limit,
    }
    try:
        items = _run_actor_sync(IG_ACTOR, payload)
    except Exception as exc:
        print(f"[apify-ig] error: {exc}")
        return []

    leads: list[dict] = []
    for it in items[:limit]:
        username = (it.get("username") or "").strip()
        if not username:
            continue
        company = (it.get("fullName") or username).strip() or username
        bio = it.get("biography") or ""
        # Apify's IG actor surfaces these fields when public:
        email = (it.get("publicEmail") or it.get("businessEmail") or "").strip()
        phone = (it.get("publicPhoneNumber") or it.get("businessPhoneNumber") or "").strip()
        website = (it.get("externalUrl") or "").strip()

        # A2.2 — recency proof. Apify's IG scraper returns latestPosts[0].timestamp
        # (ISO-8601). This rides through search_pipeline to the enricher, which
        # turns it into the social_silent signal if too old.
        latest_post_iso = ""
        latest_posts = it.get("latestPosts") or []
        if isinstance(latest_posts, list) and latest_posts:
            first = latest_posts[0]
            if isinstance(first, dict):
                latest_post_iso = (first.get("timestamp") or "").strip()
        if not latest_post_iso:
            latest_post_iso = (it.get("latestPostTimestamp") or "").strip()

        leads.append(
            {
                "name": "Owner",
                "company": company,
                "email": email,
                "website": website,
                "instagram": username,
                "facebook": "",
                "phone": phone,
                "industry": niche,
                "_bio": bio[:200],  # kept for scorer context, stripped before save
                "latest_post_iso": latest_post_iso,
                "posts_count": int(it.get("postsCount") or 0),
                "followers_count": int(it.get("followersCount") or 0),
            }
        )
    return leads


# ---- Facebook ------------------------------------------------------------- #


def _scrape_facebook(location: str, niche: str, limit: int) -> list[dict]:
    """Search Facebook pages matching the niche + location."""
    if limit <= 0:
        return []
    payload = {
        "searchQueries": [f"{niche} {location}"],
        "maxPages": limit,
    }
    try:
        items = _run_actor_sync(FB_ACTOR, payload)
    except Exception as exc:
        print(f"[apify-fb] error: {exc}")
        return []

    leads: list[dict] = []
    for it in items[:limit]:
        company = (it.get("title") or it.get("name") or "").strip()
        if not company:
            continue
        page_url = (it.get("pageUrl") or it.get("url") or "").strip()
        email = (it.get("email") or "").strip()
        phone = (it.get("phone") or it.get("phoneNumber") or "").strip()
        website = (it.get("website") or "").strip()

        # A2.2 — last-post recency from FB
        latest_post_iso = (
            it.get("latestPostDate")
            or it.get("lastPostDate")
            or it.get("lastPostTime")
            or ""
        )
        if isinstance(latest_post_iso, str):
            latest_post_iso = latest_post_iso.strip()
        else:
            latest_post_iso = ""

        leads.append(
            {
                "name": "Owner",
                "company": company,
                "email": email,
                "website": website,
                "instagram": "",
                "facebook": page_url,
                "phone": phone,
                "industry": niche,
                "latest_post_iso": latest_post_iso,
            }
        )
    return leads


# ---- Public API ----------------------------------------------------------- #


def find_businesses(location: str, niche: str, count: int = 5) -> list[dict[str, Any]]:
    """Pull leads from Instagram + Facebook via Apify.

    Splits ``count`` (capped by APIFY_MAX_RESULTS) roughly evenly between
    the two platforms. Dedupes by lowercased company name.
    """
    if not _token():
        print("[apify] APIFY_TOKEN missing — skipping")
        return []

    cap = min(count, _max_results())
    if cap <= 0:
        return []

    ig_n = (cap + 1) // 2
    fb_n = cap - ig_n

    print(f"[apify]   Searching IG ({ig_n}) + FB ({fb_n}) for '{niche}' in {location}")
    ig_leads = _scrape_instagram(location, niche, ig_n)
    fb_leads = _scrape_facebook(location, niche, fb_n)

    seen: set[str] = set()
    out: list[dict] = []
    for lead in ig_leads + fb_leads:
        key = lead["company"].lower()
        if key in seen:
            continue
        seen.add(key)
        # Strip the bio scratch field — schema doesn't carry it
        lead.pop("_bio", None)
        out.append(lead)
        if len(out) >= cap:
            break
    print(f"[apify]   IG={len(ig_leads)} FB={len(fb_leads)} → {len(out)} unique")
    return out
