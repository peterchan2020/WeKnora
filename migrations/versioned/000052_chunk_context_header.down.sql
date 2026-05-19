-- Remove context_header and child_chunk_ids columns from chunks table
ALTER TABLE chunks DROP COLUMN IF EXISTS context_header;
ALTER TABLE chunks DROP COLUMN IF EXISTS child_chunk_ids;