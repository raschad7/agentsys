-- AgentFlow database schema
--
-- Apply with the SQL editor in your Supabase project, or psql.
-- Safe to re-run on a fresh DB; for an existing DB use the ALTER block at
-- the bottom to backfill the new columns.

create table if not exists leads (
  id          uuid primary key default gen_random_uuid(),
  name        text,
  email       text unique,
  company     text,
  website     text,
  industry    text,
  -- Social + alternative contact channels (added when leads come from
  -- Tavily/Apify rather than OSM — many businesses outside the US/EU
  -- have no website and live on Instagram/Facebook).
  instagram   text,
  facebook    text,
  phone       text,
  score       int  default 0,
  -- Status values used by the pipeline:
  --   new                — fresh from finder, not yet processed
  --   qualified          — scorer said yes, ready for outreach
  --   contacted          — outreach actually sent (Instantly returned ok)
  --   rejected           — scorer said no
  --   no_contact_email   — no email available, queue for manual DM
  --   error              — pipeline failed somewhere
  --   replied            — webhook fired
  status      text default 'new',
  created_at  timestamptz default now()
);

create table if not exists outreach (
  id              uuid primary key default gen_random_uuid(),
  lead_id         uuid references leads(id) on delete cascade,
  email_subject   text,
  email_body      text,
  sent_at         timestamptz,
  opened          bool default false,
  replied         bool default false,
  follow_up_count int  default 0,
  -- Status values:
  --   pending  — row created before send attempt
  --   sent     — Instantly accepted the message (sent_at populated)
  --   failed   — send attempt errored out
  --   replied  — reply webhook fired
  status          text default 'pending',
  created_at      timestamptz default now()
);

create table if not exists agent_logs (
  id          uuid primary key default gen_random_uuid(),
  agent_name  text,
  lead_id     uuid,
  action      text,
  result      text,
  error       text,
  -- Token + cost telemetry from tools/llm.py
  model       text,
  prompt_tokens     int,
  completion_tokens int,
  total_tokens      int,
  created_at  timestamptz default now()
);

create index if not exists idx_leads_status      on leads (status);
create index if not exists idx_leads_email       on leads (email);
create index if not exists idx_outreach_lead_id  on outreach (lead_id);
create index if not exists idx_logs_lead_id      on agent_logs (lead_id);
create index if not exists idx_logs_created_at   on agent_logs (created_at desc);


-- ---------------------------------------------------------------------------
-- Migration block: run these once on existing databases that were created
-- before the social/contact + token-logging additions. Idempotent.
-- ---------------------------------------------------------------------------
alter table leads      add column if not exists instagram         text;
alter table leads      add column if not exists facebook          text;
alter table leads      add column if not exists phone             text;
alter table agent_logs add column if not exists model             text;
alter table agent_logs add column if not exists prompt_tokens     int;
alter table agent_logs add column if not exists completion_tokens int;
alter table agent_logs add column if not exists total_tokens      int;
