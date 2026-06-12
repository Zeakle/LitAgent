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
    embedding VECTOR(1536),                           -- SPECTER2 向量 (Phase 6 前用零向量占位)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),    -- 创建时间
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),    -- 最后更新时间
    UNIQUE(key)                                       -- key 唯一，防止重复知识条目
);

CREATE INDEX idx_semantic_type ON semantic_entries(entry_type);  -- 按类型筛选
CREATE INDEX idx_semantic_key ON semantic_entries(key);          -- 按 key 精确查询

-- ── Procedural Memory ──
CREATE TABLE IF NOT EXISTS procedures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),    -- 主键
    name VARCHAR(255) NOT NULL,                        -- Procedure 名称 (如 "CV extraction")
    version VARCHAR(32) NOT NULL DEFAULT '1.0.0',     -- 语义版本号
    status VARCHAR(32) NOT NULL DEFAULT 'active',      -- active | deprecated | disabled
    description TEXT NOT NULL,                         -- 描述这个 Procedure 做什么
    source_file VARCHAR(512),                          -- 对应 YAML 文件路径 (.claude/skills/extraction/cv.yaml)
    trigger_embedding VECTOR(1536),                    -- 触发词的 embedding (Phase 6)
    success_count INTEGER NOT NULL DEFAULT 0,          -- 历史成功次数
    failure_count INTEGER NOT NULL DEFAULT 0,          -- 历史失败次数
    avg_duration_ms INTEGER NOT NULL DEFAULT 0,        -- 平均执行耗时 (毫秒)
    last_executed_at TIMESTAMPTZ,                      -- 上次执行时间
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),     -- 创建时间
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),     -- 最后更新时间
    UNIQUE(name, version)                              -- 同名同版本不可重复
);

CREATE TABLE IF NOT EXISTS trigger_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),    -- 主键
    procedure_id UUID NOT NULL                        -- 关联的 Procedure
        REFERENCES procedures(id) ON DELETE CASCADE,  -- Procedure 删了，触发词自动删
    pattern VARCHAR(512) NOT NULL,                    -- 触发词文本 (如 "extract CV paper")
    locale VARCHAR(10) DEFAULT 'en',                  -- 语言: en | zh-CN
    weight REAL NOT NULL DEFAULT 1.0,                 -- 匹配权重，高频触发词设高权重
    UNIQUE(procedure_id, pattern)                     -- 同 Procedure 不重复
);

CREATE INDEX idx_triggers_procedure ON trigger_patterns(procedure_id);  -- 按 Procedure 查触发词
