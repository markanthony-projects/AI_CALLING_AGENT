-- This file no longer defines the schema. Alembic does.
--
-- It previously declared ON DELETE CASCADE on three foreign keys, ON DELETE SET NULL on a
-- fourth, and two indexes -- none of which existed in the running database, because the
-- schema had actually been built by SQLAlchemy's create_all. Anyone who ran this file got
-- a different schema from production.
--
-- To create or update the schema:
--     alembic upgrade head
--
-- Only the pgvector extension still needs to exist before the first migration runs, and it
-- requires database-owner privileges that the application user may not have.

CREATE EXTENSION IF NOT EXISTS vector;
