CREATE TABLE IF NOT EXISTS corpus_ingestion_batches (
    batch_id TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL,
    manifest_path TEXT,
    manifest_hash TEXT,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_code TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS corpus_paper_state (
    collection_name TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
    batch_id TEXT,
    asset_hash TEXT,
    metadata_hash TEXT,
    corpus_version TEXT,
    schema_version TEXT,
    parser_version TEXT,
    chunking_version TEXT,
    chunk_strategy TEXT,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    embedding_backend TEXT,
    embedding_model TEXT,
    embedding_document_adapter TEXT,
    active_point_ids TEXT[] NOT NULL DEFAULT '{}',
    chunk_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_input_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_code TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_name, paper_id)
);

ALTER TABLE corpus_paper_state
    ADD COLUMN IF NOT EXISTS chunk_strategy TEXT,
    ADD COLUMN IF NOT EXISTS chunk_size INTEGER,
    ADD COLUMN IF NOT EXISTS chunk_overlap INTEGER,
    ADD COLUMN IF NOT EXISTS embedding_backend TEXT,
    ADD COLUMN IF NOT EXISTS embedding_document_adapter TEXT,
    ADD COLUMN IF NOT EXISTS embedding_input_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS payload_hashes JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_corpus_paper_state_resume
    ON corpus_paper_state(collection_name, status);
