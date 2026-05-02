"""OpenStreetMap-based lead discovery — completely free, no API key.

Two free public APIs are used:
  - Nominatim (https://nominatim.openstreetmap.org)  → geocode "Brooklyn, NY"
  - Overpass (https://overpass-api.de)               → query businesses by tag

OSM's data quality varies a lot by region (very dense in EU/US cities,
patchy elsewhere). When a query returns nothing the caller is expected
to fall back to another source.

Both APIs have usage policies — be reasonable, set a User-Agent.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

USER_AGENT = "AgentFlow/1.0 (lead discovery; contact: noreply@agentflow.local)"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# How a free-text niche maps to OSM tag filters. Tries each combo until
# something returns results. The list is intentionally generous — OSM
# tags businesses inconsistently.
NICHE_TAGS: dict[str, list[tuple[str, str]]] = {
    "bakery":        [("shop", "bakery")],
    "restaurant":    [("amenity", "restaurant")],
    "cafe":          [("amenity", "cafe")],
    "coffee shop":   [("amenity", "cafe")],
    "coffee":        [("amenity", "cafe")],
    "bar":           [("amenity", "bar"), ("amenity", "pub")],
    "gym":           [("leisure", "fitness_centre"), ("leisure", "sports_centre")],
    "fitness":       [("leisure", "fitness_centre")],
    "yoga":          [("leisure", "fitness_centre"), ("sport", "yoga")],
    "salon":         [("shop", "hairdresser"), ("shop", "beauty")],
    "hairdresser":   [("shop", "hairdresser")],
    "barber":        [("shop", "hairdresser"), ("shop", "beauty")],
    "spa":           [("leisure", "spa"), ("shop", "beauty")],
    "dentist":       [("amenity", "dentist"), ("healthcare", "dentist")],
    "doctor":        [("amenity", "doctors"), ("healthcare", "doctor")],
    "clinic":        [("amenity", "clinic"), ("healthcare", "clinic")],
    "pharmacy":      [("amenity", "pharmacy")],
    "vet":           [("amenity", "veterinary")],
    "veterinarian":  [("amenity", "veterinary")],
    "lawyer":        [("office", "lawyer")],
    "law firm":      [("office", "lawyer")],
    "accountant":    [("office", "accountant")],
    "consultant":    [("office", "consulting")],
    "real estate":   [("office", "estate_agent")],
    "estate agent":  [("office", "estate_agent")],
    "photographer":  [("craft", "photographer")],
    "florist":       [("shop", "florist")],
    "bookstore":     [("shop", "books")],
    "books":         [("shop", "books")],
    "bicycle shop":  [("shop", "bicycle")],
    "bike shop":     [("shop", "bicycle")],
    "butcher":       [("shop", "butcher")],
    "tailor":        [("shop", "tailor"), ("craft", "tailor")],
    "jeweler":       [("shop", "jewelry")],
    "jewelry":       [("shop", "jewelry")],
    "optician":      [("shop", "optician")],
    "pet shop":      [("shop", "pet")],
    "auto repair":   [("shop", "car_repair"), ("craft", "car_repair")],
    "car repair":    [("shop", "car_repair")],
    "hotel":         [("tourism", "hotel"), ("tourism", "guest_house")],
    "guest house":   [("tourism", "guest_house")],
    "school":        [("amenity", "school")],
    "music school":  [("amenity", "music_school")],
    "art gallery":   [("tourism", "gallery"), ("amenity", "arts_centre")],
}


def _http() -> httpx.Client:
    return httpx.Client(
        timeout=30.0,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


def geocode(location: str) -> Optional[dict[str, Any]]:
    """Look up a city/region/place. Returns {lat, lon, bbox: [s, w, n, e]}."""
    params = {"q": location, "format": "json", "limit": 1, "addressdetails": 0}
    try:
        with _http() as client:
            resp = client.get(NOMINATIM_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        print(f"[osm] geocode failed: {exc}")
        return None
    if not data:
        return None
    hit = data[0]
    bbox = hit.get("boundingbox")  # [south, north, west, east] as strings
    if not bbox or len(bbox) != 4:
        return None
    s, n, w, e = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return {
        "lat": float(hit["lat"]),
        "lon": float(hit["lon"]),
        "bbox": (s, w, n, e),
        "display_name": hit.get("display_name", ""),
    }


def _tag_filters_for(niche: str) -> list[tuple[str, str]]:
    n = niche.strip().lower()
    if n in NICHE_TAGS:
        return NICHE_TAGS[n]
    # Try partial matches before giving up
    for key, tags in NICHE_TAGS.items():
        if key in n or n in key:
            return tags
    return []


def _build_query(niche: str, bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"
    filters = _tag_filters_for(niche)

    if filters:
        parts = []
        for k, v in filters:
            parts.append(f'  node["{k}"="{v}"]({bbox_str});')
            parts.append(f'  way["{k}"="{v}"]({bbox_str});')
        body = "\n".join(parts)
    else:
        # Free-text fallback: match the niche against the name field.
        # Limit scope so it doesn't return tens of thousands of hits.
        safe = re.sub(r"[^a-zA-Z0-9 ]", "", niche)
        body = (
            f'  node["name"~"{safe}",i]({bbox_str});\n'
            f'  way["name"~"{safe}",i]({bbox_str});'
        )

    return (
        f"[out:json][timeout:25];\n"
        f"(\n{body}\n);\n"
        f"out center tags 80;"
    )


def _derive_email(website: str, fallback_local: str = "info") -> str:
    if not website:
        return ""
    try:
        host = urlparse(website if "://" in website else f"http://{website}").hostname
    except Exception:
        return ""
    if not host:
        return ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    # Skip social / aggregator domains where info@ wouldn't reach the business
    blocked = {"facebook.com", "instagram.com", "twitter.com", "x.com",
               "tiktok.com", "yelp.com", "tripadvisor.com", "linktr.ee",
               "google.com", "linkedin.com"}
    if host in blocked or any(host.endswith("." + b) for b in blocked):
        return ""
    return f"{fallback_local}@{host}"


def _normalise_url(url: str) -> str:
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        return "http://" + url
    return url


def find_businesses(
    location: str,
    niche: str,
    count: int = 5,
) -> list[dict]:
    """Find real businesses in OSM by location + niche.

    Returns a list of lead dicts shaped like the rest of the pipeline expects:
        {name, company, email, website, industry}
    Best-effort: returns [] on geocode/query failure or empty results.
    """
    geo = geocode(location)
    if not geo:
        print(f"[osm] could not geocode {location!r}")
        return []

    query = _build_query(niche, geo["bbox"])
    try:
        with _http() as client:
            # Nominatim asks for >=1s between requests — we already made one.
            time.sleep(1.0)
            resp = client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        print(f"[osm] overpass query failed: {exc}")
        return []

    elements = data.get("elements", []) or []
    leads: list[dict] = []
    seen_names: set[str] = set()

    for el in elements:
        tags = el.get("tags") or {}
        name = (tags.get("name") or "").strip()
        if not name or name.lower() in seen_names:
            continue
        seen_names.add(name.lower())

        website = _normalise_url(
            tags.get("website")
            or tags.get("contact:website")
            or tags.get("url")
            or ""
        )
        email = (tags.get("email") or tags.get("contact:email") or "").strip()
        if not email:
            email = _derive_email(website)

        # Skip leads with no way to contact them — not useful for outreach.
        if not email:
            continue

        industry_label = niche
        for k in ("shop", "amenity", "leisure", "office", "craft", "healthcare", "tourism"):
            if k in tags:
                industry_label = tags[k].replace("_", " ")
                break

        leads.append(
            {
                "name": "Owner",  # OSM doesn't carry owner names
                "company": name,
                "email": email,
                "website": website,
                "industry": industry_label,
            }
        )
        if len(leads) >= count:
            break

    return leads
