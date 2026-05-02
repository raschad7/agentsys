"""Enricher agent — fetches the website and extracts real signals.

This is what makes Tier-2 scoring evidence-based instead of LLM-guess-based.
Before: scorer was given (name, company, industry, website) and asked to vibe.
Now: enricher actually loads the page and reports concrete facts:

  - Is the site alive?  HTTP status, redirects.
  - Is it slow?         Wall-clock load time, page weight.
  - Is it modern?       HTTPS, mobile viewport meta, framework hints.
  - Is it old/template? Wix / Squarespace / GoDaddy / Weebly markers.
  - Reachable contacts? <form> on page, mailto: links.
  - Social links?       Instagram/Facebook/Twitter URLs scraped from HTML.
  - About-page hint?    Title + meta description so the email can be specific.

Signal keys are kept stable — the dashboard's SIGNAL_RENDER map in index.html
references them by name, so renaming requires a frontend change too.

Best-effort: any failure is captured as the ``website_dead`` signal rather
than raising — a dead website is itself a high-value signal for our agency.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from tools import enrichment_cache, events

# Bound the parallel-enrichment fan-out. Most searches return ≤20 leads,
# spread across distinct domains (not the same one) — so 8 concurrent fetches
# finish a typical batch in ~1 slow-fetch worth of time. Above 8 you start
# hitting your own outbound bandwidth cap and DNS contention without much
# benefit.
PARALLEL_WORKERS = 8

USER_AGENT = "AgentFlow/1.0 (+https://agentflow.local)"
HTTP_TIMEOUT = 8.0
MAX_HTML_BYTES = 1_500_000          # cap on what we'll read into RAM
SLOW_SECONDS = 3.0                  # >this counts as slow_site
HEAVY_BYTES = 1_500_000             # >this counts as slow_site too


# Markers that pin down the site's tech stack. Order matters: first hit wins.
# These are case-insensitive substring searches over the raw HTML.
_TECH_MARKERS: list[tuple[str, str]] = [
    ("wix.com",           "Wix"),
    ("static.wixstatic",  "Wix"),
    ("squarespace.com",   "Squarespace"),
    ("squarespace-cdn",   "Squarespace"),
    ("weebly.com",        "Weebly"),
    ("godaddysites.com",  "GoDaddy"),
    ("godaddy",           "GoDaddy"),
    ("webflow.io",        "Webflow"),
    ("/wp-content/",      "WordPress"),
    ("/wp-includes/",     "WordPress"),
    ("cdn.shopify.com",   "Shopify"),
    ("myshopify.com",     "Shopify"),
    ("__next_data__",     "Next.js"),
    ("data-reactroot",    "React"),
    ("__nuxt",            "Nuxt"),
    ("ng-version",        "Angular"),
    ("data-vue",          "Vue"),
]
# Builders we treat as "outdated tech" for our pitch (they're cheap templates,
# good upgrade prospects). React/Next/Vue/Nuxt count as "modern_site" instead.
_OUTDATED_BUILDERS = {"Wix", "Squarespace", "Weebly", "GoDaddy"}
_MODERN_FRAMEWORKS = {"Next.js", "React", "Nuxt", "Vue", "Angular", "Webflow"}

_VIEWPORT_RE = re.compile(
    r'<meta[^>]+name=["\']viewport["\']', re.IGNORECASE
)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_DESC_RE = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_FORM_RE = re.compile(r"<form\b", re.IGNORECASE)
_MAILTO_RE = re.compile(r'href=["\']mailto:([^"\']+)["\']', re.IGNORECASE)
_TEL_RE = re.compile(r'href=["\']tel:([^"\']+)["\']', re.IGNORECASE)
_IG_RE = re.compile(
    r'https?://(?:www\.)?instagram\.com/([A-Za-z0-9_.][A-Za-z0-9_.]{0,29})',
    re.IGNORECASE,
)
_FB_RE = re.compile(
    r'https?://(?:www\.)?facebook\.com/([A-Za-z0-9_.\-]+)', re.IGNORECASE
)

# A2.1 — copyright year extraction. Matches "© 2018", "&copy; 2020-2022",
# "Copyright 2017" etc. Take the latest year encountered; if it's more than
# STALE_YEAR_GAP behind the current year, the site is presumed neglected.
_COPYRIGHT_RE = re.compile(
    r'(?:©|&copy;|copyright)\s*(?:&\#?[a-z0-9]+;)?\s*(?:&[a-z]+;\s*)?(\d{4})',
    re.IGNORECASE,
)
STALE_YEAR_GAP = 2

# A2.1 — phrases that strongly indicate the business is permanently closed.
# Lowercase-only matching; we lowercase the page text before scanning. Kept
# narrow on purpose — false positives mean we silently drop a real lead.
_CLOSED_PHRASES = (
    "permanently closed",
    "we have closed",
    "we've closed our doors",
    "we have closed our doors",
    "out of business",
    "no longer in business",
    "this business is closed",
    "going out of business",
    "closed for good",
    "ceased trading",
    "ceased operations",
)

# A2.1 — fingerprints of parking-page providers. If any of these strings is
# in the HTML, the "website" is really a placeholder; there's no business
# behind it (yet) and we shouldn't email anyone derived from this domain.
_PARKED_MARKERS = (
    "sedoparking.com",
    "parkingcrew.net",
    "bodis.com",
    "domainnamesales.com",
    "this web page is parked",
    "this domain is parked",
    "parked free, courtesy of",
    "this domain may be for sale",
    "buy this domain",
    "domain for sale",
    "sedo.com/search/details",
)

# A2.2 — IG/FB activity threshold. Days since the latest post; over this we
# treat the social presence as silent (business is dormant or shuttered).
SOCIAL_SILENT_DAYS = 180


def _normalise_url(url: str) -> str:
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url


def _days_since(iso: str) -> int | None:
    """Days between an ISO-8601 timestamp and now (UTC). None if unparseable."""
    if not iso:
        return None
    try:
        # Tolerate trailing Z and missing offset
        cleaned = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return max(0, int(delta.total_seconds() // 86400))
    except (ValueError, TypeError):
        return None


def _empty_signals() -> dict[str, bool]:
    """Default false for every key so the dashboard doesn't see undefined."""
    return {
        "no_website":          False,
        "website_dead":        False,
        "slow_site":           False,
        "outdated_tech":       False,
        "no_mobile_viewport":  False,
        "no_https":            False,
        "has_contact_form":    False,
        "active_social":       False,
        "modern_site":         False,
        # A2 reliability signals
        "business_closed":     False,    # kill switch in scorer
        "parked_domain":       False,    # kill switch in scorer
        "business_stale":      False,    # +1: copyright/footer years out of date
        "social_silent":       False,    # +1: IG/FB inactive >180d
    }


def _empty_evidence() -> dict[str, Any]:
    return {
        "signals": _empty_signals(),
        "tech": "",
        "title": "",
        "meta_description": "",
        "load_seconds": 0.0,
        "page_bytes": 0,
        "found_email": "",
        "found_phone": "",
        "found_instagram": "",
        "found_facebook": "",
    }


def _extract_signals_from_html(html: str, final_url: str, evidence: dict) -> None:
    """Mutate ``evidence`` in place with everything we can read off the page."""
    sig = evidence["signals"]

    # Modernity signals
    sig["no_https"] = not final_url.startswith("https://")
    sig["no_mobile_viewport"] = _VIEWPORT_RE.search(html) is None

    # Tech stack
    for marker, label in _TECH_MARKERS:
        if marker.lower() in html.lower():
            evidence["tech"] = label
            break
    if evidence["tech"] in _OUTDATED_BUILDERS:
        sig["outdated_tech"] = True
    elif evidence["tech"] in _MODERN_FRAMEWORKS:
        sig["modern_site"] = True

    # Slow / heavy
    if evidence["load_seconds"] > SLOW_SECONDS:
        sig["slow_site"] = True
    if evidence["page_bytes"] > HEAVY_BYTES:
        sig["slow_site"] = True

    # Content/contact
    if _FORM_RE.search(html) or _MAILTO_RE.search(html):
        sig["has_contact_form"] = True
    if not evidence["found_email"]:
        m = _MAILTO_RE.search(html)
        if m:
            evidence["found_email"] = m.group(1).split("?")[0].strip()
    if not evidence["found_phone"]:
        m = _TEL_RE.search(html)
        if m:
            evidence["found_phone"] = m.group(1).strip()

    # Social links
    ig = _IG_RE.search(html)
    if ig:
        handle = ig.group(1).lower()
        # Skip Instagram's own helper paths
        if handle not in {"p", "explore", "accounts", "directory", "reel", "tv"}:
            evidence["found_instagram"] = handle
            sig["active_social"] = True
    fb = _FB_RE.search(html)
    if fb:
        slug = fb.group(1).lower()
        if slug not in {"sharer", "dialog", "tr", "plugins"}:
            evidence["found_facebook"] = f"https://facebook.com/{slug}"
            sig["active_social"] = True

    # Title + meta description for grounding the email later
    t = _TITLE_RE.search(html)
    if t:
        evidence["title"] = re.sub(r"\s+", " ", t.group(1)).strip()[:200]
    md = _META_DESC_RE.search(html)
    if md:
        evidence["meta_description"] = md.group(1).strip()[:500]

    # ---- A2.1 reliability signals ----
    html_lower = html.lower()

    # Parked domain: explicit registrar/parking-platform fingerprints
    if any(marker in html_lower for marker in _PARKED_MARKERS):
        sig["parked_domain"] = True

    # Permanently closed: scan for explicit closure phrases. Title + body.
    haystack = (evidence["title"] + " " + html_lower)[:200_000]  # cap scan size
    for phrase in _CLOSED_PHRASES:
        if phrase in haystack:
            sig["business_closed"] = True
            evidence["closed_phrase"] = phrase
            break

    # Stale: latest copyright year in footer is >2 years old
    years = [int(y) for y in _COPYRIGHT_RE.findall(html) if 1990 <= int(y) <= 2100]
    if years:
        latest = max(years)
        evidence["latest_copyright_year"] = latest
        current = datetime.now(timezone.utc).year
        if latest <= current - STALE_YEAR_GAP:
            sig["business_stale"] = True


def _fetch_website(url: str) -> tuple[str, str, dict]:
    """Return ``(html, final_url, evidence)``. Empty html on failure."""
    evidence = _empty_evidence()
    started = time.monotonic()
    try:
        with httpx.Client(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            elapsed = time.monotonic() - started
            evidence["load_seconds"] = round(elapsed, 2)
            final_url = str(resp.url)
            if resp.status_code >= 400:
                evidence["signals"]["website_dead"] = True
                return "", final_url, evidence
            # Cap how much HTML we read — some sites send 50MB of inlined data
            html = resp.text[:MAX_HTML_BYTES]
            evidence["page_bytes"] = len(resp.content)
            return html, final_url, evidence
    except Exception as exc:
        evidence["signals"]["website_dead"] = True
        evidence["load_seconds"] = round(time.monotonic() - started, 2)
        evidence["error"] = str(exc)
        return "", url, evidence


def _apply_social_signals(lead: dict, evidence: dict) -> None:
    """Mutate evidence with active_social / social_silent based on lead meta.

    Lives separately so we can apply it to both fresh enrichments AND cached
    ones — the underlying social-recency data is per-lead, not per-domain.
    """
    sig = evidence["signals"]
    if lead.get("instagram") or lead.get("facebook"):
        sig["active_social"] = True
    days_silent = _days_since(lead.get("latest_post_iso") or "")
    if days_silent is not None:
        evidence["latest_post_age_days"] = days_silent
        if days_silent > SOCIAL_SILENT_DAYS:
            sig["social_silent"] = True
            sig["active_social"] = False


def _domain_from_url(url: str) -> str:
    """Lowercased registrable host from a URL. Empty if unparseable."""
    if not url:
        return ""
    try:
        host = urlparse(url if "://" in url else f"http://{url}").hostname
    except Exception:
        return ""
    if not host:
        return ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def enrich_lead(
    lead: dict,
    *,
    run_id: str = "",
    lead_index: int = 0,
) -> dict:
    """Fetch the lead's website (if any) and extract signals + evidence.

    Returns a dict shaped:
        {signals: {...bool}, tech, title, meta_description, load_seconds,
         page_bytes, found_email, found_phone, found_instagram, found_facebook,
         from_cache: bool}

    Always emits ``enricher_start`` and ``enricher_complete`` on the bus so
    the dashboard's pipeline view animates.
    """
    company = lead.get("company") or lead.get("name") or "(unknown)"
    website = _normalise_url(lead.get("website", "") or "")
    lead_id = lead.get("id", "")

    print(f"[ENRICH]   {company} → {website or '(no website)'}")
    events.emit(
        run_id,
        "enricher_start",
        {"index": lead_index, "lead_id": lead_id, "company": company, "website": website},
    )

    if not website:
        evidence = _empty_evidence()
        evidence["signals"]["no_website"] = True
        # Even with no website, we may still have IG meta worth checking
        _apply_social_signals(lead, evidence)
        events.emit(
            run_id,
            "enricher_complete",
            {
                "index": lead_index,
                "lead_id": lead_id,
                "signals": evidence["signals"],
                "tech": "",
                "load_seconds": 0.0,
                "from_cache": False,
            },
        )
        return evidence

    # ---- B8: cache lookup by domain ----
    domain = _domain_from_url(website)
    cached = enrichment_cache.get(domain) if domain else None
    if cached is not None:
        evidence = dict(cached)
        # Cached signals are website-derived. Re-apply per-lead social
        # signals because they depend on lead-supplied meta, not domain.
        evidence["signals"] = dict(cached.get("signals") or _empty_signals())
        _apply_social_signals(lead, evidence)
        evidence["from_cache"] = True
        print(
            f"[ENRICH]   cache HIT for {domain} "
            f"(saved {evidence.get('load_seconds', 0)}s)"
        )
        events.emit(
            run_id,
            "enricher_complete",
            {
                "index": lead_index,
                "lead_id": lead_id,
                "signals": evidence["signals"],
                "tech": evidence.get("tech", ""),
                "load_seconds": evidence.get("load_seconds", 0.0),
                "page_bytes": evidence.get("page_bytes", 0),
                "title": evidence.get("title", ""),
                "from_cache": True,
            },
        )
        return evidence

    # ---- Cache miss: fetch the page ----
    html, final_url, evidence = _fetch_website(website)
    if html:
        _extract_signals_from_html(html, final_url, evidence)

    _apply_social_signals(lead, evidence)
    evidence["from_cache"] = False

    print(
        f"[ENRICH]   tech={evidence['tech'] or '?'} "
        f"load={evidence['load_seconds']}s "
        f"signals={[k for k,v in evidence['signals'].items() if v]}"
    )

    # Cache by domain — strip the per-lead social signals before storing
    # so they don't leak between leads sharing a domain.
    if domain:
        cacheable = dict(evidence)
        cacheable["signals"] = {
            k: v for k, v in evidence["signals"].items()
            if k not in ("active_social", "social_silent")
        }
        cacheable.pop("latest_post_age_days", None)
        enrichment_cache.put(domain, cacheable)

    events.emit(
        run_id,
        "enricher_complete",
        {
            "index": lead_index,
            "lead_id": lead_id,
            "signals": evidence["signals"],
            "tech": evidence["tech"],
            "load_seconds": evidence["load_seconds"],
            "page_bytes": evidence["page_bytes"],
            "title": evidence["title"],
            "from_cache": False,
        },
    )
    return evidence


# --------------------------------------------------------------------------- #
# B7 — parallel enrichment fan-out
# --------------------------------------------------------------------------- #


def enrich_leads_parallel(
    items: list[tuple[int, dict]],
    *,
    run_id: str = "",
    max_workers: int = PARALLEL_WORKERS,
) -> dict[int, dict]:
    """Enrich multiple leads concurrently.

    ``items`` is a list of ``(lead_index, lead_dict)`` tuples. Returns a dict
    keyed by lead_index → evidence dict, in the same shape ``enrich_lead``
    returns. Per-lead failures land as a synthetic ``website_dead`` evidence
    so the pipeline still has something to score on.

    Cap on parallelism: PARALLEL_WORKERS (default 5). HTTP-bound work
    parallelises trivially with a thread pool — no asyncio plumbing needed.
    """
    results: dict[int, dict] = {}
    if not items:
        return results
    workers = min(max_workers, max(1, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_index = {
            ex.submit(enrich_lead, lead, run_id=run_id, lead_index=index): index
            for index, lead in items
        }
        for fut in as_completed(future_to_index):
            idx = future_to_index[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                # Critical: emit a synthetic enricher_complete here, otherwise
                # the dashboard card stays at "awaiting enricher…" forever.
                # The worker's enricher_start may or may not have fired before
                # the crash; emitting complete is safe either way.
                print(f"[ENRICH]   parallel worker failed for lead {idx}: {exc}")
                ev = _empty_evidence()
                ev["signals"]["website_dead"] = True
                ev["error"] = str(exc)
                results[idx] = ev
                events.emit(
                    run_id,
                    "enricher_complete",
                    {
                        "index": idx,
                        "lead_id": "",
                        "signals": ev["signals"],
                        "tech": "",
                        "load_seconds": 0.0,
                        "from_cache": False,
                        "error": str(exc),
                    },
                )
    return results
