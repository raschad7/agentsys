"""Centralised OpenAI client used by every agent.

Why this exists:
  Agents used to each instantiate their own ``OpenAI()``, hand-roll fence
  stripping, and parse JSON manually. That meant every prompt change risked
  breaking parsing in 3 different files. This module owns:

    - one OpenAI client (lazy-initialised, env-driven)
    - structured outputs via ``response_format={"type":"json_schema",...}``
      so the model returns guaranteed-shape JSON we can ``json.loads`` blindly
    - automatic retry with exponential backoff on transient errors
    - token + cost telemetry pushed to ``log_action`` so every call shows up
      in ``agent_logs`` with model, tokens, and result preview
    - a dirt-simple ``render(template, **vars)`` that replaces ``{key}``
      without choking on literal JSON braces in the prompt

Public API:

    render(template, **values)              # placeholder substitution
    call_text(model, system, user, ...)     # plain text completion
    call_json(model, system, user, schema)  # guaranteed-JSON completion

Both ``call_*`` functions return a ``LLMResult`` dataclass carrying the
parsed value plus token usage, so callers can emit it on the event bus.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from dotenv import load_dotenv

from tools.supabase_client import log_action

load_dotenv(override=True)


class BudgetExceeded(RuntimeError):
    """Raised before an LLM call would put us over the daily USD cap."""


# Per-process daily spend tracker. Resets at first call after UTC midnight.
# In-memory only — survives the lifetime of one server process. For multi-
# worker deployments swap to Redis. Documented limitation in .env.example.
_BUDGET = {"date": "", "usd": 0.0}
_BUDGET_LOCK = threading.Lock()


def _budget_cap_usd() -> float:
    raw = os.getenv("OPENAI_DAILY_USD_CAP", "").strip()
    if not raw:
        return 0.0  # 0 = no cap
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def _today() -> str:
    return date.today().isoformat()


def _check_budget() -> None:
    """Raise BudgetExceeded if today's spend has already hit the cap."""
    cap = _budget_cap_usd()
    if cap <= 0:
        return
    with _BUDGET_LOCK:
        if _BUDGET["date"] != _today():
            _BUDGET["date"] = _today()
            _BUDGET["usd"] = 0.0
        if _BUDGET["usd"] >= cap:
            raise BudgetExceeded(
                f"OpenAI daily budget cap reached: "
                f"${_BUDGET['usd']:.4f} / ${cap:.2f} (resets at UTC midnight)"
            )


def _record_spend(usd: float) -> None:
    if usd <= 0:
        return
    cap = _budget_cap_usd()
    with _BUDGET_LOCK:
        if _BUDGET["date"] != _today():
            _BUDGET["date"] = _today()
            _BUDGET["usd"] = 0.0
        _BUDGET["usd"] = round(_BUDGET["usd"] + usd, 6)
        if cap > 0 and _BUDGET["usd"] >= cap:
            print(
                f"[llm] ⚠ daily budget cap reached: "
                f"${_BUDGET['usd']:.4f} / ${cap:.2f}"
            )


def get_daily_spend() -> dict:
    """Public read-only view of today's OpenAI spend (for /api/stats)."""
    cap = _budget_cap_usd()
    with _BUDGET_LOCK:
        if _BUDGET["date"] != _today():
            return {"date": _today(), "usd": 0.0, "cap_usd": cap}
        return {"date": _BUDGET["date"], "usd": _BUDGET["usd"], "cap_usd": cap}


# Rough per-1k-token pricing (USD) so we can estimate spend in logs.
# Update these as OpenAI moves prices around — they're only for telemetry,
# nothing relies on them being exact.
_PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o":      (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
}


@dataclass
class LLMResult:
    text: str = ""
    data: Any = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


# ---- client + helpers ----------------------------------------------------- #


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = _PRICING_PER_1K.get(model, (0.0, 0.0))
    return round(prompt_tokens / 1000 * p_in + completion_tokens / 1000 * p_out, 6)


def render(template: str, **values: str) -> str:
    """Replace ``{key}`` placeholders without tripping on literal JSON braces."""
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out


# ---- retry wrapper -------------------------------------------------------- #


_RETRYABLE_HINTS = ("rate", "timeout", "connection", "overload", "503", "502", "504")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(h in msg for h in _RETRYABLE_HINTS)


def _with_retry(fn, attempts: int = 3, base_delay: float = 0.8):
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if i == attempts - 1 or not _is_retryable(exc):
                raise
            time.sleep(base_delay * (2 ** i))
    raise last  # pragma: no cover - unreachable


# ---- public entry points -------------------------------------------------- #


def call_text(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.4,
    agent: str = "llm",
    lead_id: Optional[str] = None,
    action: str = "completion",
) -> LLMResult:
    """Plain text chat completion with retry + token logging."""
    _check_budget()  # raises BudgetExceeded before the API call if over cap

    def _do():
        return _get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )

    try:
        resp = _with_retry(_do)
    except Exception as exc:
        log_action(agent, lead_id, action, "", f"llm error: {exc}", model=model)
        raise

    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    tt = getattr(usage, "total_tokens", 0) or (pt + ct)
    cost = _estimate_cost(model, pt, ct)
    _record_spend(cost)

    log_action(
        agent, lead_id, action,
        result=f"chars={len(text)} cost=${cost}",
        model=model, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
    )
    return LLMResult(text=text, model=model, prompt_tokens=pt,
                     completion_tokens=ct, total_tokens=tt, cost_usd=cost,
                     raw={"text": text})


def call_json(
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    *,
    schema_name: str = "Output",
    temperature: float = 0.4,
    agent: str = "llm",
    lead_id: Optional[str] = None,
    action: str = "completion",
) -> LLMResult:
    """Guaranteed-shape JSON completion via OpenAI structured outputs.

    ``schema`` is a JSON Schema object describing the output. OpenAI enforces
    it server-side when ``strict=True``, so the response is always valid JSON
    matching the schema — no fence stripping, no try/except json.loads dance.

    Falls back to ``response_format={"type":"json_object"}`` + manual parse if
    the chosen model doesn't support strict structured outputs.
    """
    _check_budget()  # raises BudgetExceeded before the API call if over cap

    def _do_strict():
        return _get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        )

    def _do_loose():
        return _get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system + " Output valid JSON only."},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )

    try:
        resp = _with_retry(_do_strict)
    except Exception as exc:
        # Some older models / API tiers reject strict json_schema. Try loose mode.
        msg = str(exc).lower()
        if "json_schema" in msg or "response_format" in msg or "unsupported" in msg:
            print(f"[llm] strict json_schema not accepted ({exc}); falling back")
            try:
                resp = _with_retry(_do_loose)
            except Exception as exc2:
                log_action(agent, lead_id, action, "", f"llm error: {exc2}", model=model)
                raise
        else:
            log_action(agent, lead_id, action, "", f"llm error: {exc}", model=model)
            raise

    text = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError as exc:
        log_action(agent, lead_id, action, "", f"json parse: {exc} :: {text[:200]}",
                   model=model)
        raise ValueError(f"LLM returned non-JSON despite structured output: {text!r}") from exc

    usage = getattr(resp, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) or 0
    ct = getattr(usage, "completion_tokens", 0) or 0
    tt = getattr(usage, "total_tokens", 0) or (pt + ct)
    cost = _estimate_cost(model, pt, ct)
    _record_spend(cost)

    log_action(
        agent, lead_id, action,
        result=f"json_keys={list(data.keys()) if isinstance(data, dict) else 'list'} cost=${cost}",
        model=model, prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
    )
    return LLMResult(text=text, data=data, model=model,
                     prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
                     cost_usd=cost, raw={"text": text, "data": data})
