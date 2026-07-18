-- Enable the pgvector extension (Required for the embedding column)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the custom Enum types
CREATE TYPE leadstatus AS ENUM ('HOT', 'WARM', 'COLD');
CREATE TYPE campaignstatus AS ENUM ('ACTIVE', 'PAUSED', 'COMPLETED');
CREATE TYPE callstatus AS ENUM ('IN_PROGRESS', 'COMPLETED', 'FAILED');

-- ==========================================
-- TABLE: projects
-- ==========================================
CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id VARCHAR,
    name VARCHAR NOT NULL,
    city VARCHAR NOT NULL,
    locality VARCHAR NOT NULL,
    min_price NUMERIC,
    max_price NUMERIC,
    config_json JSONB,
    amenities VARCHAR[],
    nearby_facilities JSONB,
    possession_status VARCHAR,
    usps VARCHAR[],
    rera_id VARCHAR,
    embedding VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS ix_projects_campaign_id ON projects (campaign_id);
CREATE INDEX IF NOT EXISTS ix_projects_embedding ON projects USING hnsw (embedding vector_cosine_ops);

-- ==========================================
-- TABLE: campaigns
-- ==========================================
CREATE TABLE IF NOT EXISTS campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR NOT NULL,
    status campaignstatus DEFAULT 'ACTIVE',
    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP,
    total_leads_dialed NUMERIC DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ix_campaigns_project_id ON campaigns (project_id);

-- ==========================================
-- TABLE: leads
-- ==========================================
CREATE TABLE IF NOT EXISTS leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_name VARCHAR,
    preferred_location VARCHAR,
    budget NUMERIC,
    timeline VARCHAR,
    callback_time TIMESTAMP,
    site_visit_time TIMESTAMP,
    status leadstatus
);

-- ==========================================
-- TABLE: calls
-- ==========================================
CREATE TABLE IF NOT EXISTS calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
    lead_id UUID REFERENCES leads(id) ON DELETE SET NULL,
    call_sid VARCHAR NOT NULL UNIQUE,
    status callstatus DEFAULT 'IN_PROGRESS',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    duration_seconds NUMERIC
);

CREATE INDEX IF NOT EXISTS ix_calls_call_sid ON calls (call_sid);
CREATE INDEX IF NOT EXISTS ix_calls_campaign_id ON calls (campaign_id);

-- ==========================================
-- TABLE: transcripts
-- ==========================================
CREATE TABLE IF NOT EXISTS transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id UUID NOT NULL UNIQUE REFERENCES calls(id) ON DELETE CASCADE,
    full_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
