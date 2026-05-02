"""Pre-send email validation — cheap DNS-level checks.

Why MX-record checks matter:
    Sending to a domain that has no MX record means the email will bounce
    100% of the time. Bounces poison your sender reputation faster than
    anything else; 3-5% bounce rate is enough for Gmail to start spam-folding
    your other campaigns. Catching no-MX before we hit Instantly is the
    single highest-ROI deliverability check we can add (free, ~30ms).

What this *doesn't* catch:
    - Whether the specific mailbox exists ("info@" might not be a real inbox).
      That requires SMTP RCPT-TO probing or a paid verification API
      (ZeroBounce / Hunter / NeverBounce) — Tier-C / paid work.
    - Catch-all domains that 250-OK everything regardless of mailbox.
    - Spam traps and disposable-domain lists.

For now we surface a verdict the caller can act on:
    "valid"           — has MX, looks deliverable
    "syntax_invalid"  — not a parseable address
    "no_mx"           — domain has no MX record (will 100% bounce)
    "dns_error"       — DNS lookup failed (transient — caller may want to retry)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

try:
    import dns.resolver as _dns_resolver
    import dns.exception as _dns_exception
    _HAS_DNS = True
except ImportError:  # pragma: no cover - dnspython missing => skip gracefully
    _HAS_DNS = False


# RFC-5322 simplified — good enough to reject obvious junk before DNS.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# Process-level cache so we don't re-resolve the same domain N times in a run.
# Keyed by domain → (verdict, mx_host, expires_at).
_CACHE: dict[str, tuple[str, str, float]] = {}
_CACHE_TTL_SECONDS = 60 * 60  # 1h is plenty — MX records rarely change


@dataclass
class ValidationResult:
    email: str
    verdict: str           # one of: valid | syntax_invalid | no_mx | dns_error
    mx_host: str = ""      # primary MX hostname when verdict==valid
    error: str = ""

    @property
    def deliverable(self) -> bool:
        return self.verdict == "valid"


def _split_local_domain(email: str) -> tuple[str, str]:
    if "@" not in email:
        return email, ""
    local, _, domain = email.rpartition("@")
    return local.strip(), domain.strip().lower()


def _cached(domain: str) -> Optional[tuple[str, str]]:
    hit = _CACHE.get(domain)
    if not hit:
        return None
    verdict, mx, expires = hit
    if time.monotonic() > expires:
        _CACHE.pop(domain, None)
        return None
    return verdict, mx


def _store(domain: str, verdict: str, mx: str = "") -> None:
    _CACHE[domain] = (verdict, mx, time.monotonic() + _CACHE_TTL_SECONDS)


def validate_email(email: str) -> ValidationResult:
    """Run cheap pre-send checks on ``email`` and return a verdict."""
    addr = (email or "").strip()
    if not addr or not _EMAIL_RE.match(addr):
        return ValidationResult(email=addr, verdict="syntax_invalid",
                                error="not a valid email syntax")

    _, domain = _split_local_domain(addr)
    if not domain:
        return ValidationResult(email=addr, verdict="syntax_invalid",
                                error="missing domain")

    if not _HAS_DNS:
        # dnspython not installed — return a soft-pass so the pipeline still
        # works without it, but log the warning. Production should install it.
        print("[email_validator] dnspython not installed — skipping MX check")
        return ValidationResult(email=addr, verdict="valid", mx_host="",
                                error="mx check skipped (dnspython missing)")

    cached = _cached(domain)
    if cached:
        verdict, mx = cached
        return ValidationResult(email=addr, verdict=verdict, mx_host=mx)

    try:
        # Short timeout — we'd rather fail-fast than block the dashboard.
        resolver = _dns_resolver.Resolver()
        resolver.timeout = 3.0
        resolver.lifetime = 5.0
        answers = resolver.resolve(domain, "MX")
        mx_records = sorted(
            ((int(getattr(r, "preference", 0)), str(getattr(r, "exchange", "")).rstrip("."))
             for r in answers),
            key=lambda t: t[0],
        )
        if not mx_records:
            _store(domain, "no_mx")
            return ValidationResult(email=addr, verdict="no_mx",
                                    error=f"domain {domain} has no MX records")
        mx = mx_records[0][1]
        _store(domain, "valid", mx)
        return ValidationResult(email=addr, verdict="valid", mx_host=mx)
    except _dns_resolver.NoAnswer:
        _store(domain, "no_mx")
        return ValidationResult(email=addr, verdict="no_mx",
                                error=f"domain {domain} returned no MX answer")
    except _dns_resolver.NXDOMAIN:
        _store(domain, "no_mx")
        return ValidationResult(email=addr, verdict="no_mx",
                                error=f"domain {domain} does not exist")
    except (_dns_exception.Timeout, _dns_exception.DNSException) as exc:
        # Don't cache transient errors — caller can retry.
        return ValidationResult(email=addr, verdict="dns_error",
                                error=f"DNS lookup failed: {exc}")
