"""Conditional edge functions for the LangGraph pipeline."""

from langgraph.graph import END


def route_after_scorer(state: dict) -> str:
    """score >= 3 → outreach, otherwise END."""
    if state.get("error"):
        return END
    if int(state.get("score", 0)) >= 3:
        return "outreach"
    return END


def route_after_outreach(state: dict) -> str:
    """Always terminates — success and error both end the pipeline."""
    return END
