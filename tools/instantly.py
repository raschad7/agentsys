"""Instantly.ai email sending wrapper.

Tries the Instantly v2 API first (Bearer auth — what the modern dashboard
issues keys for). Falls back to v1 (api_key in JSON body). If neither is
configured, prints the email to console so the rest of the pipeline can be
exercised without any external service.

Env vars:
    INSTANTLY_API_KEY            — required to attempt a real send
    INSTANTLY_CAMPAIGN_ID        — v2 only: UUID of the campaign to add the lead to
    INSTANTLY_API_VERSION        — optional: "v1" | "v2" | "auto" (default: auto)
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

INSTANTLY_API_KEY = os.getenv("INSTANTLY_API_KEY", "").strip()
INSTANTLY_CAMPAIGN_ID = os.getenv("INSTANTLY_CAMPAIGN_ID", "").strip()
INSTANTLY_API_VERSION = os.getenv("INSTANTLY_API_VERSION", "auto").strip().lower()

V2_LEADS_URL = "https://api.instantly.ai/api/v2/leads"
V1_SEND_URL = "https://api.instantly.ai/api/v1/send-email"


def _looks_like_v2_key(key: str) -> bool:
    """v2 keys are base64-ish with `==` padding, much longer than v1."""
    return len(key) > 60 and key.endswith("==")


def _send_v2(to_email: str, subject: str, body: str) -> dict[str, Any]:
    """Add a lead to an Instantly campaign so the configured campaign sends it."""
    if not INSTANTLY_CAMPAIGN_ID:
        return {
            "success": False,
            "mode": "v2",
            "error": (
                "INSTANTLY_CAMPAIGN_ID is not set — v2 needs a campaign UUID to "
                "deliver the email. Add INSTANTLY_CAMPAIGN_ID=<uuid> to .env or "
                "set INSTANTLY_API_VERSION=v1."
            ),
        }
    payload = {
        "campaign": INSTANTLY_CAMPAIGN_ID,
        "email": to_email,
        "personalization": {"subject": subject, "body": body},
        "custom_variables": {"subject": subject, "body": body},
    }
    headers = {
        "Authorization": f"Bearer {INSTANTLY_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(V2_LEADS_URL, json=payload, headers=headers)
            resp.raise_for_status()
            return {"success": True, "mode": "v2", "response": resp.json()}
    except httpx.HTTPStatusError as exc:
        return {
            "success": False,
            "mode": "v2",
            "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
        }
    except Exception as exc:
        return {"success": False, "mode": "v2", "error": str(exc)}


def _send_v1(to_email: str, subject: str, body: str) -> dict[str, Any]:
    payload = {
        "api_key": INSTANTLY_API_KEY,
        "to": to_email,
        "subject": subject,
        "body": body,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(V1_SEND_URL, json=payload)
            resp.raise_for_status()
            return {"success": True, "mode": "v1", "response": resp.json()}
    except httpx.HTTPStatusError as exc:
        return {
            "success": False,
            "mode": "v1",
            "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}",
        }
    except Exception as exc:
        return {"success": False, "mode": "v1", "error": str(exc)}


def send_email(to_email: str, subject: str, body: str) -> dict[str, Any]:
    """Send an email. Returns {success, mode, response|error}."""
    if not INSTANTLY_API_KEY:
        print(f"[instantly] (dry-run) to={to_email}")
        print(f"[instantly] subject: {subject}")
        print(f"[instantly] body:\n{body}")
        return {"success": True, "mode": "dryrun"}

    version = INSTANTLY_API_VERSION
    if version == "auto":
        version = "v2" if _looks_like_v2_key(INSTANTLY_API_KEY) else "v1"

    if version == "v2":
        return _send_v2(to_email, subject, body)
    return _send_v1(to_email, subject, body)
