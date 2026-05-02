"""Tests for the AgentFlow pipeline (Tier 2 — evidence-based scoring).

Mocks ``tools.llm.call_json`` and uses the in-memory Supabase fallback
(forced via conftest) so the suite runs offline.

After Tier 2:
  - The scorer's score comes from deterministic rules over signals.
  - The LLM only writes the reasoning sentence.
  - Tests therefore drive the score via the ``signals`` field in state,
    not via the LLM mock.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents import lead_scorer, outreach
from graph.pipeline import ClientState, run_pipeline
from tools import llm, supabase_client


# ---- helpers -------------------------------------------------------------- #


def _result(data: dict) -> llm.LLMResult:
    return llm.LLMResult(
        text="", data=data, model="gpt-test",
        prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.0,
    )


@pytest.fixture(autouse=True)
def _reset_memory_store():
    supabase_client._memory["leads"].clear()
    supabase_client._memory["outreach"].clear()
    supabase_client._memory["agent_logs"].clear()
    yield


def _base_lead(**overrides):
    lead = {
        "id": "lead-1",
        "name": "Sara",
        "email": "sara@example.com",
        "company": "Sara's Bakery",
        "website": "http://sarasbakery.com",
        "industry": "bakery",
        "score": 0,
        "status": "new",
    }
    lead.update(overrides)
    supabase_client._memory["leads"][lead["id"]] = lead
    return lead


# ---- Deterministic rule layer (no LLM, no mocks) ------------------------- #


def test_rule_layer_no_website_floors_at_qualified():
    """no_website alone is worth +3 → score >= 3 → qualified."""
    score, pos, neg = lead_scorer._score_from_signals({"no_website": True})
    assert score == 3
    assert any("no website" in p.lower() for p in pos)


def test_rule_layer_modern_site_is_rejected():
    """A modern site clamps at 0 even with active social."""
    score, pos, neg = lead_scorer._score_from_signals({
        "modern_site": True,
        "active_social": True,  # +1
    })
    # modern_site is -2, active_social is +1 → -1 → clamp 0
    assert score == 0


def test_rule_layer_signals_stack_to_qualified():
    score, _, _ = lead_scorer._score_from_signals({
        "outdated_tech": True,
        "slow_site": True,
        "no_mobile_viewport": True,
    })
    assert score == 3  # 1+1+1


# ---- Scorer agent (rule + LLM reasoning) --------------------------------- #


def test_scorer_uses_signals_to_score_and_llm_for_reasoning():
    lead = _base_lead()
    fake = _result({"reasoning": "Site is on Wix and slow — easy upgrade."})
    with patch.object(lead_scorer.llm, "call_json", return_value=fake):
        state: ClientState = {
            "lead_id": lead["id"],
            "name": lead["name"],
            "company": lead["company"],
            "website": lead["website"],
            "signals": {"outdated_tech": True, "slow_site": True, "no_mobile_viewport": True},
            "enrichment": {"tech": "Wix", "load_seconds": 4.2},
        }
        out = lead_scorer.lead_scorer_agent(state)
    assert out["score"] == 3
    assert out["status"] == "qualified"
    assert "Wix" in out["reasoning"] or "slow" in out["reasoning"].lower()


def test_scorer_rejects_when_modern_site():
    lead = _base_lead()
    fake = _result({"reasoning": "Modern stack, low urgency."})
    with patch.object(lead_scorer.llm, "call_json", return_value=fake):
        state: ClientState = {
            "lead_id": lead["id"],
            "name": lead["name"],
            "company": lead["company"],
            "signals": {"modern_site": True},
        }
        out = lead_scorer.lead_scorer_agent(state)
    assert out["score"] == 0
    assert out["status"] == "rejected"


def test_scorer_llm_failure_is_non_fatal():
    """When the LLM call raises, the deterministic score still stands and
    we fall back to a stitched-from-signals reasoning sentence."""
    lead = _base_lead()
    with patch.object(lead_scorer.llm, "call_json", side_effect=ValueError("boom")):
        state: ClientState = {
            "lead_id": lead["id"],
            "name": lead["name"],
            "company": lead["company"],
            "signals": {"no_website": True},  # +3 → qualified
        }
        out = lead_scorer.lead_scorer_agent(state)
    assert out["score"] == 3
    assert out["status"] == "qualified"
    assert out["reasoning"]                       # fallback sentence populated
    assert out.get("scorer_warning")              # warning recorded
    assert "error" not in out                     # but not a fatal error


# ---- End-to-end via legacy run_pipeline --------------------------------- #


def test_high_score_lead_runs_outreach_end_to_end():
    """run_pipeline (legacy LangGraph) feeds signals through scorer → outreach.

    The scorer asks the LLM once (reasoning), then outreach asks once (email).
    side_effect supplies both payloads in order.
    """
    lead = _base_lead()
    reasoning_payload = _result({"reasoning": "No website — strong fit."})
    email_payload = _result({
        "subject": "Your bakery online",
        "body": "Your bakery smells amazing.\nWe build simple sites.\nOpen to chatting?",
    })
    # Inject the qualifying signal directly into the lead so run_pipeline
    # carries it into state before invoking the scorer node.
    lead["signals"] = {"no_website": True}
    with patch.object(llm, "call_json",
                      side_effect=[reasoning_payload, email_payload]) as call:
        final = run_pipeline(lead)

    assert call.call_count == 2
    assert final["score"] == 3
    assert final["status"] == "contacted"
    assert final["outreach_sent"] is True
    assert final["email_subject"] == "Your bakery online"
    sent_row = supabase_client._memory["outreach"][0]
    assert sent_row["status"] == "sent"
    assert sent_row["sent_at"], "sent_at must populate when status='sent'"


def test_low_score_lead_never_reaches_outreach():
    """Modern site → score 0 → routed to END after scorer; only one LLM call."""
    lead = _base_lead(name="Modern Co", company="Modern Co")
    lead["signals"] = {"modern_site": True}
    reasoning_payload = _result({"reasoning": "Already on a modern stack."})
    with patch.object(llm, "call_json", return_value=reasoning_payload) as call:
        final = run_pipeline(lead)

    assert call.call_count == 1, "outreach must not be reached when score < 3"
    assert final["score"] == 0
    assert final["status"] == "rejected"
    assert final.get("outreach_sent") is False
    assert supabase_client._memory["outreach"] == []


# ---- Centralised LLM client ---------------------------------------------- #


def test_render_replaces_placeholders_without_format_collisions():
    """Critical: prompts contain literal JSON braces — render must not choke."""
    template = 'Hello {name}, return {"score": <0-5>}'
    out = llm.render(template, name="Mo")
    assert out == 'Hello Mo, return {"score": <0-5>}'


def test_render_handles_missing_keys_gracefully():
    template = "Hi {name}, your industry is {industry}"
    out = llm.render(template, name="Mo")
    assert "Mo" in out
    assert "{industry}" in out
