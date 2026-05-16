-- Add context_header and child_chunk_ids columns to chunks table
-- context_header: stores heading breadcrumb for semantic positioning (was ephemeral, now persisted)
-- child_chunk_ids: reverse link from parent chunks to their children (was missing)
DO $$ BEGIN RAISE NOTICE '[Migration 000043] Adding context_header and child_chunk_ids to chunks...'; END $$;

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context_header TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS child_chunk_ids JSONB;

DO $$ BEGIN RAISE NOTICE '[Migration 000043] context_header and child_chunk_ids columns ready'; END $$;