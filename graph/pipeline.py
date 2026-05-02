"""LangGraph pipeline definition and helpers."""

from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START

from agents.lead_scorer import lead_scorer_agent
from agents.outreach import outreach_agent
from graph.router import route_after_outreach, route_after_scorer
from tools import events


class ClientState(TypedDict, total=False):
    lead_id: str
    run_id: str
    lead_index: int  # position in a multi-lead search run (0 for single-lead)
    name: str
    email: str
    company: str
    website: str
    industry: str
    instagram: str
    facebook: str
    phone: str
    # Tier 2 — evidence from enricher, fed into scorer
    signals: dict       # {no_website: bool, slow_site: bool, ...}
    enrichment: dict    # {tech, title, meta_description, load_seconds, ...}
    reasoning: str      # one-sentence grounded justification from scorer
    score: int
    status: str  # new / qualified / rejected / contacted / no_contact_email / error
    outreach_sent: bool
    email_subject: str
    email_body: str
    follow_up_count: int
    error: str


def build_graph():
    """Construct the 2-node StateGraph: scorer → (maybe) outreach."""
    graph = StateGraph(ClientState)
    graph.add_node("scorer", lead_scorer_agent)
    graph.add_node("outreach", outreach_agent)

    graph.add_edge(START, "scorer")
    graph.add_conditional_edges("scorer", route_after_scorer)
    graph.add_conditional_edges("outreach", route_after_outreach)

    return graph.compile()


_compiled = None


def _get_compiled():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def _banner(lead: dict) -> None:
    bar = "━" * 29
    print(bar)
    print(f"Processing: {lead.get('name') or lead.get('company') or '(unknown)'}")
    print(f"Email: {lead.get('email', '')}")
    print(bar)


def run_pipeline(
    lead: dict,
    run_id: Optional[str] = None,
    lead_index: int = 0,
) -> dict:
    """Run one lead through the pipeline.

    ``run_id``     — bus key so agents publish live events the dashboard can render.
    ``lead_index`` — which card in the dashboard grid this lead belongs to.
    """
    _banner(lead)
    initial: dict = {
        "lead_id": lead.get("id", ""),
        "run_id": run_id or "",
        "lead_index": lead_index,
        "name": lead.get("name", "") or "",
        "email": lead.get("email", "") or "",
        "company": lead.get("company", "") or "",
        "website": lead.get("website", "") or "",
        "industry": lead.get("industry", "") or "",
        "instagram": lead.get("instagram", "") or "",
        "facebook": lead.get("facebook", "") or "",
        "phone": lead.get("phone", "") or "",
        # Tier 2 — pass through any pre-computed evidence; otherwise the
        # legacy auto-flow has no signals and the rule layer will return 0.
        "signals": lead.get("signals") or {},
        "enrichment": lead.get("enrichment") or {},
        "score": 0,
        "status": lead.get("status", "new") or "new",
        "outreach_sent": False,
        "email_subject": "",
        "email_body": "",
        "follow_up_count": 0,
        "error": "",
    }
    events.emit(
        run_id,
        "pipeline_start",
        {
            "lead_id": initial["lead_id"],
            "name": initial["name"],
            "company": initial["company"],
            "email": initial["email"],
            "website": initial["website"],
            "instagram": initial["instagram"],
            "facebook": initial["facebook"],
            "phone": initial["phone"],
            "index": lead_index,
        },
    )
    try:
        final = _get_compiled().invoke(initial)
    except Exception as exc:
        print(f"[pipeline] fatal error: {exc}")
        initial["error"] = str(exc)
        initial["status"] = "error"
        final = initial
        events.emit(run_id, "pipeline_error", {"error": str(exc), "index": lead_index})
    print("━" * 29)
    print()
    events.emit(
        run_id,
        "pipeline_done",
        {
            "score": final.get("score"),
            "status": final.get("status"),
            "outreach_sent": final.get("outreach_sent", False),
            "email_subject": final.get("email_subject", ""),
            "email_body": final.get("email_body", ""),
            "error": final.get("error", ""),
            "index": lead_index,
        },
    )
    return final
