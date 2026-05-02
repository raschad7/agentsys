# AgentFlow — AI Sales Pipeline

A multi-agent system that finds, enriches, qualifies, and reaches out to small business leads — with a live dashboard for human-in-the-loop control.

## How it works

```
Lead Finder  →  Enricher  →  Scorer  →  [you review]  →  Scribe  →  Postie
 OSM/Tavily     Web scrape   Rules+LLM   Dashboard        Drafts     Sends
 Apify social   Signals       0-5 score   per-lead        cold email  via Instantly
```

- **Find**: searches OpenStreetMap, Tavily web search, and Apify (Instagram + Facebook)
- **Enrich**: scrapes each lead's website for signals (closed business, parked domain, stale content, social links)
- **Score**: deterministic rules + LLM reasoning → 0-5 score; score ≥ 3 = qualified
- **Review**: you see every lead card live; click **✍ Draft** then **📮 Send**
- **Send**: MX validation → ZeroBounce check → Instantly delivery

---

## Prerequisites

- Python 3.11+
- A Supabase project (free tier works)
- An OpenAI API key

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/raschad7/agentsys.git
cd agentsys
pip install -r requirements.txt
```

### 2. Create `.env`

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
# Required
OPENAI_API_KEY=sk-...

# Database (falls back to in-memory if not set)
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJ...              # service_role key from project settings

# Lead sources (at least one recommended)
TAVILY_API_KEY=tvly-...          # https://app.tavily.com
APIFY_TOKEN=apify_api_...        # https://console.apify.com/account/integrations

# Email delivery (dry-run/log-only without these)
INSTANTLY_API_KEY=...
INSTANTLY_CAMPAIGN_ID=...        # UUID from Instantly campaign settings
INSTANTLY_FROM_EMAIL=you@yourdomain.com

# Email verification (recommended to protect sender reputation)
ZEROBOUNCE_API_KEY=...

# Dashboard auth (auto-generated on first run; set this for a stable key)
AGENTFLOW_API_KEY=your-secret-key

# Optional guardrails
OPENAI_DAILY_USD_CAP=1.00        # stop LLM calls when daily spend hits this
APIFY_MAX_RESULTS=20             # hard cap on Apify results per search
```

### 3. Apply the database schema

Open your Supabase project → **SQL Editor** → paste and run the contents of `schema.sql`.

### 4. Start the server

```bash
python main.py serve
```

Open **http://localhost:8000** in your browser.

---

## Using the dashboard

1. Enter **Location** (e.g. `Ramallah, Palestine`) and **Niche** (e.g. `dental clinic`)
2. Set how many leads to find and click **Go**
3. Watch cards appear live as leads are found → enriched → scored
4. Qualified leads (score ≥ 3) show a **✍ Draft email** button
5. Review the draft, edit if needed, then click **📮 Send**

### Lead statuses

| Status | Meaning |
|---|---|
| `qualified` | Score ≥ 3, awaiting your review |
| `drafted` | Email written, ready to send |
| `contacted` | Email sent via Instantly |
| `rejected` | Score < 3 or manually rejected |
| `no_contact_email` | No email found — needs manual DM |
| `error` | MX/ZeroBounce blocked the send |

### Bulk actions

- **✍ Draft all qualified** — drafts emails for every qualified lead at once
- **📮 Send all drafts** — sends all pending drafts

---

## API reference

All `/api/*` endpoints require header `X-API-Key: <AGENTFLOW_API_KEY>`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search` | Start a search (`location`, `niche`, `count`, `campaign_id`) |
| `GET` | `/api/stream/{run_id}` | SSE stream of live pipeline events |
| `POST` | `/api/draft/{lead_id}` | Write a cold-email draft |
| `POST` | `/api/send/{lead_id}` | Send the existing draft |
| `POST` | `/api/reject/{lead_id}` | Reject a lead |
| `GET` | `/api/lead/{lead_id}` | Get lead + draft details |
| `GET` | `/api/stats` | Daily spend, lead counts, cache stats |
| `GET/POST` | `/api/campaigns` | List / create campaigns |
| `PATCH` | `/api/campaigns/{id}` | Update campaign |
| `DELETE` | `/api/campaigns/{id}` | Deactivate campaign |
| `GET` | `/api/health` | Health check — `{"status":"running"}` |

---

## Lead sources

| Source key | What it searches | API key needed |
|---|---|---|
| `osm` | OpenStreetMap (free, no key) | — |
| `tavily` | Web search + GPT extraction | `TAVILY_API_KEY` |
| `apify` | Instagram + Facebook profiles | `APIFY_TOKEN` |
| `social` | Tavily first, then Apify | both |
| `auto` | OSM → Tavily → Apify (all three) | both |

Set `LEAD_SOURCE=auto` (or leave unset) to try all sources.

---

## Campaigns

Campaigns let you run different niches with custom scoring weights and pitch angles:

1. Click **+ NEW** in the dashboard
2. Set a name, ICP description, pitch angle, and score threshold
3. Select the campaign before clicking **Go**

---

## CLI commands

```bash
python main.py serve                  # start the dashboard server
python main.py import leads.csv       # import and run leads from CSV
python main.py run <lead_id>          # re-run a single lead by id
```

CSV columns: `name, email, company, website, industry`

---

## Running tests

```bash
pytest tests/ -v
```

All tests run fully offline — no real API calls, no Supabase, no ZeroBounce.

---

## Project structure

```
agentflow/
├── agents/
│   ├── lead_finder.py      Scout: OSM + Tavily + Apify
│   ├── enricher.py         Parallel website scraping + signal detection
│   ├── lead_scorer.py      Rules + LLM scoring
│   └── outreach.py         Scribe (draft) + Postie (send)
├── api/
│   └── webhooks.py         FastAPI app: dashboard, SSE stream, all /api/* routes
├── graph/
│   └── search_pipeline.py  Orchestrates finder → enricher → scorer
├── tools/
│   ├── supabase_client.py  DB helpers + in-memory fallback
│   ├── llm.py              OpenAI client with retry + spend tracking
│   ├── events.py           In-process SSE event bus
│   ├── campaigns.py        Campaign CRUD
│   ├── enrichment_cache.py 24h SQLite cache for enrichment results
│   ├── zerobounce.py       Email verification with 7-day SQLite cache
│   ├── email_validator.py  MX record check (dnspython)
│   ├── instantly.py        Instantly email delivery
│   ├── tavily.py           Tavily web search
│   ├── apify_social.py     Apify Instagram + Facebook scraping
│   └── osm.py              OpenStreetMap Nominatim search
├── prompts/                LLM prompt templates
├── web/
│   └── index.html          Single-file dashboard
├── main.py                 CLI entry point
├── requirements.txt
├── schema.sql              Supabase table definitions
└── .env.example            All supported env vars with comments
```
