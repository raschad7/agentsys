"""AgentFlow CLI entry point.

Commands:
    python main.py serve
    python main.py import leads.csv
    python main.py run <lead_id>
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Force stdout/stderr to UTF-8 so prints with arrows, bullets, or Arabic
# company names don't crash on Windows where the default is cp1252/cp1256.
# Must run before anything imports a module that prints at import time.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv

# override=True so .env always beats stale shell env vars
load_dotenv(override=True)


def _require_openai() -> None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: OPENAI_API_KEY is required. Add it to .env and try again.")
        sys.exit(1)
    # Print a fingerprint so you can confirm at a glance which key is active.
    # (Shows first 10 + last 4 chars — never the full secret.)
    fp = f"{key[:10]}…{key[-4:]}" if len(key) > 14 else "(short)"
    print(f"[env] OPENAI_API_KEY loaded: {fp}")
    instantly = os.getenv("INSTANTLY_API_KEY", "")
    if instantly:
        ifp = f"{instantly[:8]}…{instantly[-4:]}" if len(instantly) > 12 else "(short)"
        print(f"[env] INSTANTLY_API_KEY loaded: {ifp}")
    else:
        print("[env] INSTANTLY_API_KEY: (not set — emails will dry-run)")


def cmd_serve() -> None:
    _require_openai()
    import uvicorn

    uvicorn.run("api.webhooks:app", host="0.0.0.0", port=8000, reload=False)


def cmd_import(csv_path: str) -> None:
    _require_openai()
    from graph.pipeline import run_pipeline
    from tools.supabase_client import insert_lead

    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: file not found: {csv_path}")
        sys.exit(1)

    qualified = 0
    rejected = 0
    sent = 0

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    print(f"Importing {len(rows)} leads from {csv_path}\n")

    for raw in rows:
        lead = {
            "name": (raw.get("name") or "").strip() or None,
            "email": (raw.get("email") or "").strip() or None,
            "company": (raw.get("company") or "").strip() or None,
            "website": (raw.get("website") or "").strip() or None,
            "industry": (raw.get("industry") or "").strip() or None,
        }
        try:
            stored = insert_lead(lead)
        except Exception as exc:
            print(f"[import] skipped {lead.get('email')}: {exc}")
            continue

        final = run_pipeline(stored)
        status = final.get("status")
        if status == "contacted":
            qualified += 1
            sent += 1
        elif status == "qualified":
            qualified += 1
        elif status == "rejected":
            rejected += 1

    print("DONE")
    print(f"Qualified: {qualified} | Rejected: {rejected} | Emails sent: {sent}")


def cmd_run(lead_id: str) -> None:
    _require_openai()
    from graph.pipeline import run_pipeline
    from tools.supabase_client import get_lead

    lead = get_lead(lead_id)
    if not lead:
        print(f"ERROR: lead not found: {lead_id}")
        sys.exit(1)
    final = run_pipeline(lead)
    print("DONE")
    print(
        f"Score: {final.get('score')} | Status: {final.get('status')} | "
        f"Sent: {final.get('outreach_sent', False)}"
    )


def _usage() -> None:
    print(__doc__ or "")
    print("Usage:")
    print("  python main.py serve")
    print("  python main.py import <csv_path>")
    print("  python main.py run <lead_id>")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        _usage()
        sys.exit(1)
    cmd = argv[1]
    if cmd == "serve":
        cmd_serve()
    elif cmd == "import":
        if len(argv) < 3:
            print("ERROR: import requires a CSV path")
            sys.exit(1)
        cmd_import(argv[2])
    elif cmd == "run":
        if len(argv) < 3:
            print("ERROR: run requires a lead_id")
            sys.exit(1)
        cmd_run(argv[2])
    else:
        _usage()
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
