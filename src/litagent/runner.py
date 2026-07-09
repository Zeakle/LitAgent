"""LitAgent runner — 装配层，将 AppConfig 串接为可运行的 Survey Pipeline。

用法:
    config = load_config()
    async with LitAgent(config) as agent:
        report = await agent.run("few-shot learning in CV")

三级降级:
    Fatal      — LLM Client / ToolRegistry / ToolExecutor 不可用 → ConfigError
    Degradable — Redis / Qdrant / PostgreSQL 不可用 → warn + 跳过对应功能

LangFuse trace 预留:
    LitAgent(config, trace_hook=my_tracer) → Phase 13 注入 langfuse.observe()
"""

from __future__ import annotations
import uuid
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from litagent.config import AppConfig
from litagent.exceptions import ConfigError
from litagent.llm.client import BaseLLMClient, OpenAICompatibleClient
from litagent.safety.budget import CostBudget
from litagent.safety.injection import InjectionDetector
from litagent.tools.registry import get_registry
from litagent.tools.executor import ToolExecutor
from litagent.tools.builtin.search import register_search_tools
from litagent.tools.builtin.extract import register_extract_tools
from litagent.context.budget import BudgetManager
from litagent.mcp.bridge import MCPBridge
from litagent.skills.manager import SkillManager
from litagent.agents.extraction_strategy import (
    RegexStrategy,
    LLMStrategy,
    ResilientExtractionStrategy
)
from litagent.agents.search import SearchWorker
from litagent.agents.extractor import ExtractorWorker
from litagent.agents.dedup import DedupWorker
from litagent.agents.graph import GraphWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.report import ReportWorker
from litagent.agents.planner import SurveyPlanner
from litagent.orchestrator.scheduler import Scheduler, Worker
from litagent.orchestrator.message_bus import MessageBus
from litagent.orchestrator.task_graph import TaskGraph
from litagent.logging import get_logger, setup_logging

# ── Memory infrastructure ──
from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.manager import MemoryManager

# ── RAG infrastructure ──
from litagent.rag.vector_store import QdrantVectorStore
from litagent.rag.claims_index import ClaimsIndex
from litagent.rag.reranker import CrossEncoderReranker
from litagent.rag.retriever import HybridRetriever

# ── Observation ──
from litagent.observability.tracing import LangFuseTracer


logger = get_logger('runner')


# ═══════════════════════════════════════════════════════════
# Infra — 可选基础设施组件
# ═══════════════════════════════════════════════════════════


@dataclass
class Infra:
    memory: MemoryManager | None = None
    claims_index: ClaimsIndex | None = None
    retriever: HybridRetriever | None = None

    _qdrant_client: Any = None
    _redis_client: Any = None
    _pg_pool: Any = None


# ═══════════════════════════════════════════════════════════
# Trace hook 类型
# ═══════════════════════════════════════════════════════════


TraceHook = Callable[[str, dict[str, Any]], Any]
"""Trace 回调签名: (event_name: str, event_data: dict) -> Any

event_name 取值:
    "survey.start"     — run() 开始
    "survey.complete"  — run() 完成
    "survey.error"     — run() 异常
    "wire.start"       — _wire() 开始
    "wire.complete"    — _wire() 完成
    "cleanup.start"    — cleanup() 开始
    "cleanup.complete" — cleanup() 完成
"""


# ═══════════════════════════════════════════════════════════
# LitAgent — 装配类
# ═══════════════════════════════════════════════════════════


class LitAgent:
    """LitAgent 装配类 —— 从 AppConfig 串接所有组件，运行文献综述。

    三步生命周期:
        1. __init__  → 保存 config，不做 I/O
        2. _wire()   → 按拓扑序创建所有组件（__aenter__ 调用）
        3. cleanup() → 关闭连接 + consolidate（__aexit__ 调用）

    用法:
        config = load_config()
        async with LitAgent(config) as agent:
            report = await agent.run("survey few-shot learning in CV")

    Args:
        config: AppConfig 实例
        trace_hook: Phase 13 LangFuse 集成预留
    """


    def __init__(self, config: AppConfig, trace_hook: TraceHook | None = None) -> None:
        self._config = config
        self._session_id = str(uuid.uuid4())[:8]
        self._trace_hook = trace_hook

        # ── Essential (Fatal 级) ──
        self._cost_budget: CostBudget | None = None
        self._llm: BaseLLMClient | None = None
        self._registry = None
        self._executor: ToolExecutor | None = None
        self._budget_manager: BudgetManager | None = None

        self._infra: Infra = Infra()

        # ── Skills + Strategies ──
        self._skill_manager: SkillManager | None = None
        self._extraction_strategy: ResilientExtractionStrategy | None = None
        self._mcp_bridge = None

        # ── Workers (8) ──
        self._search: SearchWorker | None = None
        self._dedup: DedupWorker | None = None
        self._extractor: ExtractorWorker | None = None
        self._graph: GraphWorker | None = None
        self._synthesis: SynthesisWorker | None = None
        self._reviewer: ReviewerWorker | None = None
        self._adversarial: AdversarialReviewWorker | None = None
        self._report: ReportWorker | None = None

        # ── Orchestrator ──
        self._bus: MessageBus | None = None
        self._scheduler: Scheduler | None = None
        self._planner: SurveyPlanner | None = None

        self._wired = False


    async def __aenter__(self) -> "LitAgent":
        await self._wire()
        return self

    
    async def __aexit__(self, *args: Any) -> None:
        await self.cleanup()

    
     # ── Run ──

    async def run(self, query: str) -> dict[str, Any]:
        """运行一次完整的文献综述。

        Returns:
            dict: {survey, metadata, review_history, graph_data, partial}
        """
        if not self._wired:
            await self._wire()

        self._emit('survey.start', {'query': query})
        logger.info(f'Starting survey: {query}')

        try:
            graph = await self._planner.plan(query)
            results = await self._scheduler.run(graph)
            report_data = self._extract_report(results, query)
            report_data['partial'] = (
                self._cost_budget.is_exceeded() if self._cost_budget else False
            )

            self._emit("survey.complete", {
                "query": query,
                "rounds": report_data.get("metadata", {}).get("total_rounds", 0),
                "accepted": report_data.get("metadata", {}).get("accepted", False),
            })
            logger.info("Survey complete: %d chars", len(report_data.get("survey", "")))
            return report_data

        except Exception as e:
            self._emit('survey.error', {'query': query, 'error': str(e)})
            raise
        
    
    def _extract_report(self, results: dict[str, Any], query: str) -> dict[str, Any]:
        """从 Scheduler 的 {task_id: result} 中提取最终报告。"""
        survey_text = ''
        review_history: list[dict] = []
        graph_data: dict[str, Any] = {}
        metadata: dict[str, Any] = {
            "query": query,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_rounds": 0,
            "final_score": 0.0,
            "accepted": False,
        }

        for task_id, result in results.items():
            if not isinstance(result, dict):
                continue

            if 'final_draft' in result:
                survey_text = result.get("final_draft", "")
                metadata["total_rounds"] = result.get("total_rounds", 0)
                metadata["final_score"] = result.get("final_score", 0.0)
                metadata["accepted"] = result.get("accepted", False)
                if result.get("rounds"):
                    review_history = result["rounds"]

            if 'nodes' in result or 'edges' in result:
                graph_data = result

            if 'draft' in result and 'final_draft' not in result:
                if not survey_text:
                    survey_text = result.get("draft", "")
        
        if not survey_text:
            survey_text = f"Survey incomplete. Partial results from {len(results)} tasks."

        return {
            "survey": survey_text,
            "metadata": metadata,
            "review_history": review_history,
            "graph_data": graph_data,
        }



    async def _wire(self) -> None:
        """按拓扑排序创建组件，失败抛ConfigError, Degradable降级"""
        if self._wired:
            return

        cfg = self._config
        setup_logging(cfg.logging)
        self._emit('wire.start', {'session_id': self._session_id})

        logger.info("Wiring LitAgent components (session %s)...", self._session_id)

        # 0. Observability（enabled 且用户没自己注入 hook → 建 LangFuseTracer）
        obs = getattr(cfg, 'observability', None)
        if obs and obs.enabled and self._trace_hook is None:
            self._trace_hook = LangFuseTracer(
                host=obs.langfuse_host,
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            )

        # 1. CostBudget
        self._cost_budget = CostBudget(
            max_tokens=cfg.safety.max_cost_tokens,
            warn_ratio=cfg.safety.cost_warn_ratio
        )

        # 2. LLM Client
        try:
            self._llm = OpenAICompatibleClient(
                base_url=cfg.llm.base_url,
                model=cfg.llm.model,
                max_tokens=cfg.llm.max_tokens,
                temperature=cfg.llm.temperature,
                cost_budget=self._cost_budget,
                trace_hook=self._trace_hook
            )
            logger.info(f"LLM client: {cfg.llm.model} @ {cfg.llm.base_url}")
        except Exception as e:
            raise ConfigError(f"Failed to create LLM client: {e}") from e

        # 3. Tool Registry + Built-in Tool
        try:
            self._registry = get_registry()
            register_search_tools()
            register_extract_tools()
            logger.info("ToolRegistry: %d tools registered", len(self._registry))
        except Exception as e:
            raise ConfigError(f"Failed to register tools: {e}") from e

        # 4. Tool Executor
        self._executor = ToolExecutor(
            registry=self._registry,
            cb_fail_threshold=cfg.resilience.cb_fail_threshold,
            cb_cooldown_seconds=cfg.resilience.cb_cooldown_seconds,
            trace_hook=self._trace_hook
        )

        # 5. BudgetManager
        self._budget_manager = BudgetManager(
            max_tokens=cfg.context.max_tokens,
            compact_threshold=cfg.context.compact_threshold,
        )

        # 6. Infrastructure
        self._infra = await self._connect_infra(cfg)

        # 7. Skills + Extraction Strategies
        skills_dir = str(Path(__file__).resolve().parent / 'skills' / 'extraction')
        self._skill_manager = SkillManager(skills_dir=skills_dir)
        if cfg.mcp_servers:
            self._mcp_bridge = MCPBridge()
            try:
                registered = await self._mcp_bridge.connect_all(cfg.mcp_servers)
                logger.info("MCP: %d tools registered", len(registered))
            except Exception as e:
                logger.warning(f"MCP bridge failed, continuing without MCP tools: {e}")
                self._mcp_bridge = None

        regex_strategy = RegexStrategy(self._executor)
        llm_strategy = LLMStrategy(self._llm, self._skill_manager)
        self._extraction_strategy = ResilientExtractionStrategy(llm_strategy, regex_strategy)

        # 8. Workers
        workers: list[Worker] = []

        self._search = SearchWorker(executor=self._executor, retriever=self._infra.retriever)
        workers.append(self._search)

        self._dedup = DedupWorker()
        workers.append(self._dedup)

        self._extractor = ExtractorWorker(
            strategy=self._extraction_strategy,
            claims_index=self._infra.claims_index,
            max_concurrent=cfg.extractor.max_concurrent,
            detector=InjectionDetector(),
        )
        workers.append(self._extractor)

        self._graph = GraphWorker()
        workers.append(self._graph)

        self._synthesis = SynthesisWorker(
            llm=self._llm, memory=self._infra.memory, budget=self._budget_manager)
        workers.append(self._synthesis)

        self._reviewer = ReviewerWorker(
            llm=self._llm, claims_index=self._infra.claims_index, budget=self._budget_manager)
        workers.append(self._reviewer)

        self._adversarial = AdversarialReviewWorker(
            llm=self._llm,
            synthesis=self._synthesis,
            reviewer=self._reviewer,
            max_rounds=cfg.adversarial.max_rounds,
            pass_threshold=cfg.adversarial.pass_threshold,
        )
        workers.append(self._adversarial)

        self._report = ReportWorker()
        workers.append(self._report)

        logger.info("Workers: %d registered", len(workers))

        # 9. MessageBus + Scheduler + Planner
        self._bus = MessageBus()
        self._scheduler = Scheduler(
            workers=workers,
            max_concurrent=cfg.orchestrator.max_concurrent,
            timeout_ms=cfg.orchestrator.timeout_ms,
            on_complete=self._on_session_complete,
            bus=self._bus,
            cost_budget=self._cost_budget,
            trace_hook=self._trace_hook
        )

        self._planner = SurveyPlanner()
        self._wired = True

        self._emit("wire.complete", {
            "session_id": self._session_id,
            "worker_count": len(workers),
        })
        logger.info("LitAgent wiring complete (session %s)", self._session_id)


    # ── Session lifecycle ──

    async def _on_session_complete(self, graph: TaskGraph) -> None:
        """Scheduler 完成回调 — consolidate session 到 Episodic/Semantic。""" 
        if not self._infra.memory:
            return

        try:
            results = graph.get_results()
            survey_text = ""
            for result in results.values():
                if isinstance(result, dict) and 'final_draft' in result:
                    survey_text = result['final_draft']
                    break
            
            if not survey_text:
                return
            
            state = {
                "messages": [
                    {"type": "human", "content": f"Survey session {self._session_id}"},
                    {"type": "ai", "content": survey_text[:4000]},
                ]
            }
            await self._infra.memory.working.set(self._session_id, state)
            await self._infra.memory.consolidate(self._session_id, llm=self._llm)
            logger.info("Session %s consolidated", self._session_id)
        except Exception as e:
            logger.warning("Session consolidation skipped: %s", e)


    # ── Infrastructure connection ──

    async def _connect_infra(self, cfg: AppConfig) -> Infra:
        """连接基础设施，失败组件设为 None。

        共享连接:
          - 单一 AsyncQdrantClient → EpisodicMemory + ClaimsIndex + QdrantVectorStore
          - 单一 asyncpg.Pool → SemanticMemory + ProceduralMemory
        """
        mem_cfg= cfg.memory
        infra = Infra()
        dim = self._get_embedding_dim()

        # ── Redis → WorkingMemory ──
        try:
            working = await WorkingMemory.connect(mem_cfg)
            infra._redis_client = working._redis
            logger.info("Infra: Redis connected (Working Memory)")
        except Exception as e:
            logger.warning("Infra: Redis unavailable — Working Memory disabled (%s)", e)
            working = None

        # ── Qdrant → Episodic + Claims + VectorStore ──
        qdrant_client = None
        episodic = None
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            qdrant_client = AsyncQdrantClient(url=mem_cfg.qdrant_url)

            # 确保三个collection存在
            for coll_name in ['episodes', 'claims', 'papers']:
                try:
                    await qdrant_client.get_collection(coll_name)
                except Exception:
                    await qdrant_client.create_collection(
                        collection_name=coll_name,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
                    logger.info("Infra: Created Qdrant collection '%s'", coll_name)
            
            episodic = EpisodicMemory(qdrant_client)
            claims_index = ClaimsIndex(qdrant_client, trace_hook=self._trace_hook)
            vector_store = QdrantVectorStore(qdrant_client, 'papers')

            infra._qdrant_client = qdrant_client
            infra.claims_index = claims_index

            reranker = CrossEncoderReranker()
            infra.retriever = HybridRetriever(vector_store, reranker, trace_hook=self._trace_hook)

            logger.info("Infra: Qdrant connected (Episodic + Claims + Papers)")
        except Exception as e:
            logger.warning("Infra: Qdrant unavailable — RAG/Claims/Episodic disabled (%s)", e)
            if qdrant_client:
                try:
                    await qdrant_client.close()
                except Exception:
                    pass

        # ── PostgreSQL → Semantic + Procedural ──
        pg_pool = None
        semantic = None
        procedural = None
        try:
            import asyncpg
            pg_pool = await asyncpg.create_pool(mem_cfg.pg_url, min_size=2, max_size=10)
            semantic = SemanticMemory(pg_pool)
            procedural = ProceduralMemory(pg_pool)
            infra._pg_pool = pg_pool
            logger.info("Infra: PostgreSQL connected (Semantic + Procedural)")
        except Exception as e:
            logger.warning("Infra: PostgreSQL unavailable — Semantic/Procedural disabled (%s)", e)

        # ── MemoryManager ──
        if working is not None and episodic is not None:
            infra.memory = MemoryManager(
                working=working, episodic=episodic,
                semantic=semantic, procedural=procedural,
                trace_hook=self._trace_hook,
            )
            logger.info('Infra: Memory Manager assembled (4-layer)')
        elif working is not None:
            logger.warning("Infra: MemoryManager skipped — Episodic unavailable (Qdrant)")
        else:
            logger.warning("Infra: MemoryManager skipped — Working unavailable (Redis)")

        return infra
        

    @staticmethod
    def _get_embedding_dim() -> int:
        from litagent.rag.embedder import get_embedder
        return get_embedder().dim


    # ── Cleanup ──

    async def cleanup(self) -> None:
        """关闭所有连接，释放资源"""
        self._emit('cleanup.start', {'session_id': self._session_id})
        logger.info('Cleaning up LitAgent (session %s)...', self._session_id)

        if self._trace_hook and hasattr(self._trace_hook, 'flush'):
            try:
                self._trace_hook.flush()
            except Exception as e:
                logger.debug(f"Tracer flush error {e}")

        if self._mcp_bridge:
            try:
                await self._mcp_bridge.disconnect_all()
            except Exception as e:
                logger.debug(f"MCP disconnect error: {e}")

        infra = self._infra

        if infra.memory:
            for backend in ['working', 'episodic', 'semantic', 'procedural']:
                b = getattr(infra.memory, backend, None)
                if b:
                    try:
                        await b.close()
                    except Exception as e:
                        logger.debug(f"{backend} closed error {e}")

        if infra.claims_index:
            try:
                await infra.claims_index.close()
            except Exception as e:
                logger.debug(f"ClaimsIndex close: {e}")

        if infra.retriever:
            try:
                await infra.retriever._store.close()
            except Exception as e:
                logger.debug(f"AectorStore (retriever) close: {e}")

        if infra._pg_pool:
            try:
                await infra._pg_pool.close()
            except Exception as e:
                logger.debug(f"PG pool close: {e}")

        if infra._qdrant_client:
            try:
                await infra._qdrant_client.close()
            except Exception as e:
                logger.debug(f"Qdrant client close: {e}")

        if infra._redis_client:
            try:
                await infra._redis_client.close()
            except Exception as e:
                logger.debug(f"Redis close: {e}")
        
        self._wired = False
        self._emit('cleanup.complete', {'session_id': self._session_id})
        logger.info('Litagent cleanup complete')


    # ── Trace hook ──

    def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """触发 trace hook"""
        if self._trace_hook:
            try:
                self._trace_hook(event, data or {})
            except Exception as e:
                logger.debug("Trace hook failed for event '%s': %s", event, e)