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
    embedding_model TEXT,
    active_point_ids TEXT[] NOT NULL DEFAULT '{}',
    chunk_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_code TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_name, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_corpus_paper_state_resume
    ON corpus_paper_state(collection_name, status);