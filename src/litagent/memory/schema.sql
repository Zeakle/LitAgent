-- Phase 5: Semantic + Procedural Memory schema
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Semantic Memory ──
CREATE TABLE IF NOT EXISTS semantic_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),    -- 主键，自动生成 ID
    entry_type VARCHAR(32) NOT NULL,                  -- 类型: user_preference | domain_knowledge | entity_relation | proven_solution
    key VARCHAR(255) NOT NULL,                        -- 唯一键，同 key 触发冲突解决 + ON CONFLICT upsert
    value JSONB NOT NULL,                             -- 存储的值，JSONB 支持索引和查询
    confidence REAL NOT NULL DEFAULT 0.5,             -- 置信度 0-1，跨多次观察提升
    source VARCHAR(32) NOT NULL DEFAULT 'inferred',   -- 来源: explicit_user | observed | inferred
    source_episode_ids TEXT[] DEFAULT '{}',           -- 从哪些 Episode 推导出来，可溯源
    embedding VECTOR(384),                           -- all-MiniLM-L6-v2 向量 (384d)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),    -- 创建时间
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),    -- 最后更新时间
    UNIQUE(key)                                       -- key 唯一，防止重复知识条目
);

CREATE INDEX idx_semantic_type ON semantic_entries(entry_type);  -- 按类型筛选
CREATE INDEX idx_semantic_key ON semantic_entries(key);          -- 按 key 精确查询

-- ── Procedural Memory (13.7.1) ──
CREATE TABLE IF NOT EXISTS procedural_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_type VARCHAR(64) NOT NULL,
    profile_key VARCHAR(255) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    scope VARCHAR(64) NOT NULL DEFAULT 'global',
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    empty_result_count INTEGER NOT NULL DEFAULT 0,
    rate_limit_count INTEGER NOT NULL DEFAULT 0,
    timeout_count INTEGER NOT NULL DEFAULT 0,
    execution_count INTEGER NOT NULL DEFAULT 0,
    avg_duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    avg_result_count REAL NOT NULL DEFAULT 0.0,
    last_executed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(profile_type, profile_key, scope)
);

CREATE INDEX IF NOT EXISTS idx_profiles_type_scope
    ON procedural_profiles(profile_type, scope);

-- schema.sql 追加（HNSW 索引）
CREATE INDEX IF NOT EXISTS idx_semantic_hnsw
    ON semantic_entries USING hnsw (embedding vector_cosine_ops)
    WITH (M = 16, ef_construction = 200);
