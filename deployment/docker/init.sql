-- ATM-Net++ PostgreSQL initialization
-- Creates extensions needed by the application

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create indexes hint (actual tables created by SQLAlchemy)
-- This file runs once on first container start
