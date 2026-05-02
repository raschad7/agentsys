"""Pytest setup: make sure tests never hit real external services.

Because `tools/*.py` now call `load_dotenv(override=True)` so user .env keys
beat stale shell env vars, we can't rely on os.environ overrides alone —
`.env` would win. Instead we import the modules once and zero out their
module-level keys so every agent run behaves like dry-run.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["AGENTFLOW_API_KEY"] = "test-auth-key"

# Import and neuter external integrations. Must happen after sys.path tweak.
from tools import instantly as _instantly  # noqa: E402
from tools import supabase_client as _sb  # noqa: E402
from tools import zerobounce as _zb  # noqa: E402

_instantly.INSTANTLY_API_KEY = ""
_instantly.INSTANTLY_CAMPAIGN_ID = ""

# Force in-memory mode regardless of what .env has set.
_sb._SUPABASE_URL = None
_sb._SUPABASE_KEY = None
_sb._client = None

# B10 — neuter ZeroBounce at function level. We can't rely on env clearing
# because load_dotenv(override=True) elsewhere will reload the real key.
# Replace verify_email with a soft-pass so Postie unit tests don't hit the
# live ZB API (and don't burn user credits).
def _fake_zb_verify(email, *, use_cache=True):
    return _zb.VerificationResult(
        email=email, verdict="pass", status="valid", error="(test bypass)",
    )
_zb.verify_email = _fake_zb_verify
