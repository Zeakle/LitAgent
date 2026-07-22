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
import asyncio
import uuid
import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from litagent.config import AppConfig
from litagent.eval.base import CTX_EVIDENCE
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
    ExtractionStrategy,
    RegexStrategy,
    LLMStrategy,
    ResilientExtractionStrategy
)
from litagent.agents.search import SearchWorker
from litagent.agents.recall import RecallWorker
from litagent.agents.extractor import ExtractorWorker
from litagent.agents.dedup import DedupWorker
from litagent.agents.relevance_gate import RelevanceGateWorker
from litagent.agents.graph import GraphWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.planner import SurveyPlanner
from litagent.orchestrator.scheduler import Scheduler, Worker
from litagent.orchestrator.task_graph import TaskGraph
from litagent.logging import get_logger, setup_logging

# ── Memory infrastructure ──
from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.manager import MemoryManager

# ── RAG infrastructure ──
from litagent.rag.interfaces import Reranker
from litagent.rag.vector_store import QdrantVectorStore
from litagent.rag.claims_index import ClaimsIndex
from litagent.rag.reranker import CrossEncoderReranker
from litagent.rag.retriever import HybridRetriever

# ── Observation ──
from litagent.observability.tracing import LangFuseTracer
from litagent.observability.recorder import RedactingTraceHook
from litagent.observability.context import set_task_id, reset_task_id

# ── Evaluation ──
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator

from litagent.evidence import collect_ledger, format_ledger, find_unknown_refs


logger = get_logger('runner')


def derive_delivery(partial: bool, quality: dict[str, Any] | None) -> dict[str, Any]:
    """partial + quality → 交付语义的唯一集中映射（纯函数）。

    partial=执行中断；quality=评估结论；accepted=reviewer 结论——三者独立，
    本函数只读不回写。partial 优先级最高（结果本身不完整，谈不上质量放行）。
    """
    q_status = (quality or {}).get('status', 'unverified')

    reason_codes: list[str] = []
    if partial:
        reason_codes.append('partial_execution')
    if q_status == 'failed':
        reason_codes.append('quality_failed')
    elif q_status == 'unverified':
        reason_codes.append('quality_unverified')

    if partial:
        status = 'partial'
    elif q_status == 'failed':
        status = 'blocked'
    elif q_status == 'unverified':
        status = 'needs_review'
    else:
        status = 'ready'

    return {
        'status': status,
        'publishable': status == 'ready',
        'reason_codes': reason_codes,
    }


# ═══════════════════════════════════════════════════════════
# Infra — 可选基础设施组件
# ═══════════════════════════════════════════════════════════


@dataclass
class Infra:
    memory: MemoryManager | None = None
    claims_index: ClaimsIndex | None = None
    retriever: HybridRetriever | None = None
    reranker: Reranker | None = None

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
        self._extraction_strategy: ExtractionStrategy | None = None
        self._mcp_bridge = None

        # ── Workers ──
        self._search: SearchWorker | None = None
        self._dedup: DedupWorker | None = None
        self._relevance_gate: RelevanceGateWorker | None = None
        self._extractor: ExtractorWorker | None = None
        self._graph: GraphWorker | None = None
        self._synthesis: SynthesisWorker | None = None
        self._reviewer: ReviewerWorker | None = None
        self._adversarial: AdversarialReviewWorker | None = None

        # ── Orchestrator ──
        self._scheduler: Scheduler | None = None
        self._planner: SurveyPlanner | None = None

        # ── Evaluator ──
        self._evaluators = []

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

        self._emit('survey.start', {'query': query, 'session_id': self._session_id})
        logger.info(f'Starting survey: {query}')

        try:
            graph = await self._planner.plan(query)
            if hasattr(self._trace_hook, "capture_graph"):
                self._trace_hook.capture_graph(graph)
            results = await self._scheduler.run(graph)
            if hasattr(self._trace_hook, "capture_graph_state"):
                self._trace_hook.capture_graph_state(graph)
            report_data = self._extract_report(results, query)
            execution = self._derive_execution(
                graph,
                budget_exceeded=(
                    self._cost_budget.is_exceeded() if self._cost_budget else False
                ),
                final_output_present=bool(
                    isinstance(results.get("adversarial_review"), dict)
                    and results["adversarial_review"].get("final_draft")
                ),
            )
            report_data["partial"] = execution["partial"]
            report_data.setdefault("metadata", {})["execution"] = execution
            report_data['evaluation'] = await self._evaluate(report_data['survey'], results)
            quality = self._derive_quality(report_data.get('evaluation', {}))
            report_data['quality'] = quality

            if quality['status'] == 'failed' and await self._attempt_evidence_rewrite(report_data, results):
                report_data['evaluation'] = await self._evaluate(report_data['survey'], results)
                quality = self._derive_quality(report_data['evaluation'])
                report_data['quality'] = quality

            report_data['delivery'] = derive_delivery(report_data['partial'], quality)

            self._emit("survey.complete", {
                "query": query,
                "rounds": report_data.get("metadata", {}).get("total_rounds", 0),
                "accepted": report_data.get("metadata", {}).get("accepted", False),
                'quality_status': quality['status'],
                'total_tokens': self._cost_budget.used if self._cost_budget else 0,
                'delivery_status': report_data['delivery']['status']
            })


            logger.info("Survey complete: %d chars", len(report_data.get("survey", "")))
            return report_data

        except Exception as e:
            self._emit('survey.error', {'query': query, 'error': str(e)})
            raise


    @staticmethod
    def _derive_execution(
        graph: TaskGraph,
        budget_exceeded: bool,
        final_output_present: bool,
    ) -> dict[str, Any]:
        summary = graph.execution_summary()
        reason_codes: list[str] = []
        if budget_exceeded:
            reason_codes.append("cost_budget_exceeded")
        if summary["failed_task_ids"]:
            reason_codes.append("task_failed")
        if summary["cancelled_task_ids"]:
            reason_codes.append("task_cancelled")
        if summary["skipped_task_ids"]:
            reason_codes.append("task_skipped")
        if not final_output_present:
            reason_codes.append("final_output_missing")

        partial = bool(reason_codes)
        return {
            **summary,
            "status": "incomplete" if partial else "complete",
            "partial": partial,
            "reason_codes": reason_codes,
        }
        
    
    def _extract_report(self, results: dict[str, Any], query: str) -> dict[str, Any]:
        """从 Scheduler 的 {task_id: result} 中按 DAG 契约提取最终报告。

        信任 DAG 的固定 task_id 作为契约——直接键查找，不遍历、不 duck-typing。
        The adversarial review task is the single successful report source.
        """
        graph_result = results.get('graph_analysis')
        graph_data = graph_result if isinstance(graph_result, dict) else {}

        base_metadata: dict[str, Any] = {
            "query": query,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_rounds": 0,
            "final_score": 0.0,
            "accepted": False,
        }

        adv_result = results.get('adversarial_review')
        if isinstance(adv_result, dict) and adv_result.get('final_draft'):
            metadata = {**base_metadata}
            metadata["generated_at"] = time.time()
            metadata["total_rounds"] = adv_result.get("total_rounds", 0)
            metadata["final_score"] = adv_result.get("final_score", 0.0)
            metadata["accepted"] = adv_result.get("accepted", False)
            return {
                "survey": adv_result["final_draft"],
                "metadata": metadata,
                "review_history": self._format_review_history(adv_result.get("rounds", [])),
                "graph_data": graph_data,
            }

        return {
            "survey": f"Survey incomplete. Partial results from {len(results)} tasks.",
            "metadata": base_metadata,
            "review_history": [],
            "graph_data": graph_data,
        }

    @staticmethod
    def _format_review_history(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        history = []
        for item in rounds:
            review = item.get("review", {})
            history.append({
                "round": item.get("round", 0),
                "score": review.get("score", 0),
                "verdict": review.get("verdict", "unknown"),
                "weaknesses": review.get("weaknesses", []),
                "issue_count": len(review.get("issues", [])),
            })
        return history


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
            langfuse = LangFuseTracer(
                host=obs.langfuse_host,
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            )
            self._trace_hook = RedactingTraceHook(
                langfuse, payload_mode=obs.payload_mode,
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
        skills_dir = str(Path(__file__).resolve().parent / 'skills')
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
        if cfg.extractor.enable_llm:
            llm_strategy = LLMStrategy(self._llm, self._skill_manager)
            self._extraction_strategy = ResilientExtractionStrategy(
                llm_strategy,
                regex_strategy,
                per_paper_timeout_ms=cfg.extractor.per_paper_timeout_ms,
            )
        else:
            self._extraction_strategy = regex_strategy

        # 8. Workers
        workers: list[Worker] = []

        self._search = SearchWorker(executor=self._executor, memory_manager=self._infra.memory)
        workers.append(self._search)

        self._recall = RecallWorker(retriever=self._infra.retriever)
        workers.append(self._recall)

        self._dedup = DedupWorker()
        workers.append(self._dedup)

        self._relevance_gate = RelevanceGateWorker(
            reranker=self._infra.reranker,
            max_papers=cfg.extractor.max_papers,
        )
        workers.append(self._relevance_gate)

        self._extractor = ExtractorWorker(
            strategy=self._extraction_strategy,
            claims_index=self._infra.claims_index,
            max_concurrent=cfg.extractor.max_concurrent,
            detector=InjectionDetector(),
            max_papers=cfg.extractor.max_papers,
        )
        workers.append(self._extractor)

        self._graph = GraphWorker(max_papers=cfg.extractor.max_papers)
        workers.append(self._graph)

        self._synthesis = SynthesisWorker(
            llm=self._llm,
            memory=self._infra.memory,
            budget=self._budget_manager,
            skill_manager=self._skill_manager,
            agent_config=cfg.agent,
        )
        workers.append(self._synthesis)

        self._reviewer = ReviewerWorker(
            llm=self._llm, claims_index=self._infra.claims_index, budget=self._budget_manager, skill_manager=self._skill_manager, config=cfg.adversarial)
        workers.append(self._reviewer)

        self._adversarial = AdversarialReviewWorker(
            llm=self._llm,
            synthesis=self._synthesis,
            reviewer=self._reviewer,
            max_rounds=cfg.adversarial.max_rounds,
            pass_threshold=cfg.adversarial.pass_threshold,
            trace_hook=self._trace_hook,
        )
        workers.append(self._adversarial)

        logger.info("Workers: %d registered", len(workers))

        # 9. Scheduler + Planner
        self._scheduler = Scheduler(
            workers=workers,
            max_concurrent=cfg.orchestrator.max_concurrent,
            timeout_ms=cfg.orchestrator.timeout_ms,
            on_complete=self._on_session_complete,
            cost_budget=self._cost_budget,
            trace_hook=self._trace_hook
        )

        self._planner = SurveyPlanner(
            llm=self._llm,
            config=cfg.planner,
            trace_hook=self._trace_hook,
            memory_manager=self._infra.memory
        )
        self._wired = True

        # 10. Evaluators（max_tokens 从 eval config 取——评估任务额度需求大）
        eval_mt = cfg.eval.max_tokens
        self._evaluators = [
            CitationEvaluator(self._llm, max_tokens=eval_mt),
            ConsistencyEvaluator(self._llm, max_tokens=eval_mt),
            RagasFaithfulnessEvaluator(self._config, llm=self._llm)
        ]

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

            # consolidate 在所有 worker 之后跑（on_complete），无 parent worker span。
            # 包一层 subspan（parent 兜底到 root），让其 llm.call 归到独立节点而非扁平挂 root。
            self._emit('subspan.start', {
                'task_id': 'consolidate', 'parent_task_id': '',
                'name': 'consolidate', 'round': 0,
            })
            token = set_task_id('consolidate')
            try:
                await self._infra.memory.consolidate(self._session_id, llm=self._llm)
            finally:
                reset_task_id(token)
                self._emit('subspan.end', {'task_id': 'consolidate'})
            logger.info("Session %s consolidated", self._session_id)
        except Exception as e:
            logger.warning("Session consolidation skipped: %s", e)


    # ── Infrastructure connection ──

    @staticmethod
    def _create_reranker() -> Reranker | None:
        try:
            return CrossEncoderReranker()
        except Exception as exc:
            logger.warning(
                f'CrossEncoder unavailable reason=model_init_failed error_type = {type(exc).__name__}'
            )

            return None

    async def _connect_infra(self, cfg: AppConfig) -> Infra:
        """连接基础设施，失败组件设为 None。

        共享连接:
          - 单一 AsyncQdrantClient → EpisodicMemory + ClaimsIndex + QdrantVectorStore
          - 单一 asyncpg.Pool → SemanticMemory + ProceduralMemory
        """
        mem_cfg= cfg.memory
        infra = Infra()

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
            dim = self._get_embedding_dim()

            # 确保三个collection存在
            for coll_name in ['episodes', 'claims']:
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
            infra._qdrant_client = qdrant_client
            infra.claims_index = claims_index

            # papers is a dual-index collection. Preserve Claims/Episodic if a
            # non-empty legacy papers collection requires an explicit migration.
            try:
                vector_store = await QdrantVectorStore.ensure_compatible(
                    qdrant_client, 'papers', dim
                )
                infra.reranker = self._create_reranker()
                infra.retriever = HybridRetriever(
                    vector_store, infra.reranker, trace_hook=self._trace_hook
                )
                logger.info("Infra: Qdrant connected (Episodic + Claims + Papers)")
            except ConfigError as e:
                logger.warning("Infra: RAG disabled pending papers schema migration (%s)", e)
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
            await procedural.ensure_tables()
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

    # ── Evaluation ──
    async def _evaluate(self, survey: str, results: dict[str, Any]) -> dict[str, Any]:
        """Evaluate -> return:{metric_name: {score, passed, skipped, details}"""
        if not self._evaluators:
            return {}

        contexts = self._build_eval_context(results)

        self._emit('subspan.start', {
            'task_id': 'evaluation',
            'parent_task_id': '',
            'name': 'evaluation',
            'round': 0
        })
        token = set_task_id('evaluation')
        out: dict[str, Any] = {}

        try:
            evals = await asyncio.gather(
                *[e.evaluate(survey, contexts) for e in self._evaluators],
                return_exceptions=True
            )

            for ev, result in zip(self._evaluators, evals):
                if isinstance(result, Exception):
                    logger.warning(f"Evaluator {ev.metric_name} raised: {result}")
                    continue
                out[result.metric] = {
                    'score': result.score,
                    'passed': result.passed,
                    'skipped': result.skipped,
                    'details': result.details
                }
        except Exception as e:
            logger.warning(f"Evaluation phase failed {e}")
        finally:
            reset_task_id(token)
            # 带 out → evaluation span 显示各评估器 {score/passed/skipped/details}，供 LangFuse 检视
            self._emit('subspan.end', {'task_id': 'evaluation', 'output': out})
        return out


    @staticmethod
    def _collect_extractions(results: dict[str, Any]) -> list[dict]:
        for result in results.values():
            if isinstance(result, list) and result and isinstance(result[0], dict) and 'claims' in result[0]:
                return result
        return []


    def _build_eval_context(self, results: dict[str, Any]) -> dict[str, Any]:
        """从 results 里 extractor 的 extractions 组装评估 context。"""
        from litagent.eval.base import CTX_PAPERS, CTX_CLAIMS
        extractions: list = []
        for result in results.values():
            if isinstance(result, list) and result and isinstance(result[0], dict) and 'claims' in result[0]:
                extractions = result
                break

        claims_texts = [c for ext in extractions for c in ext.get('claims', [])]
        return {
            CTX_PAPERS: extractions, 
            CTX_CLAIMS: [{'text': t} for t in claims_texts],
            CTX_EVIDENCE: collect_ledger(extractions)
        }


    async def _attempt_evidence_rewrite(
        self, report_data: dict[str, Any], results: dict[str, Any]
    ) -> bool:
        """quality failed 时的一次性证据修复——至多一次，预算独立于对抗轮次。"""
        existing = report_data.get("metadata", {}).get("evidence_rewrite")
        if existing and existing.get("attempted"):
            return False

        rewrite_meta: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "reason": "",
        }
        report_data.setdefault("metadata", {})["evidence_rewrite"] = rewrite_meta

        diagnostics = (
            report_data.get("evaluation", {})
            .get("faithfulness", {})
            .get("details", {})
            .get("unsupported_claims")
        )
        if not diagnostics:
            rewrite_meta["reason"] = "no_unsupported_claims"
            return False

        ledger = collect_ledger(self._collect_extractions(results))
        if not ledger:
            rewrite_meta["reason"] = "empty_evidence_ledger"
            return False

        rewrite_meta["attempted"] = True
        messages = self._synthesis.rewrite_with_evidence(
            report_data["survey"],
            format_ledger(ledger, max_chars=20000),
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
        )

        accepted = False
        reason = "llm_call_failed"
        error_type: str | None = None
        new_draft = ""
        self._emit("subspan.start", {
            "task_id": "evidence_rewrite",
            "parent_task_id": "",
            "name": "evidence_rewrite",
            "round": 0,
        })
        token = set_task_id("evidence_rewrite")

        try:
            response = await self._llm.chat(
                messages, max_tokens=self._config.eval.max_tokens
            )
            new_draft = (response.content or "").strip()
            if not new_draft:
                reason = "empty_draft"
            elif len(new_draft) < 0.5 * len(report_data["survey"]):
                reason = "draft_too_short"
            elif find_unknown_refs(new_draft, ledger):
                reason = "unknown_evidence_refs"
            else:
                accepted = True
                reason = "accepted"
        except asyncio.CancelledError:
            reason = "cancelled"
            error_type = "CancelledError"
            raise
        except Exception as exc:
            reason = "llm_call_failed"
            error_type = type(exc).__name__
        finally:
            rewrite_meta["accepted"] = accepted
            rewrite_meta["reason"] = reason
            if error_type is not None:
                rewrite_meta["error_type"] = error_type
            reset_task_id(token)
            trace_output: dict[str, Any] = {
                "accepted": accepted,
                "reason": reason,
            }
            if error_type is not None:
                trace_output["error_type"] = error_type
            self._emit("subspan.end", {
                "task_id": "evidence_rewrite",
                "output": trace_output,
            })

        if not accepted:
            logger.warning("Evidence rewrite rejected: %s", reason)
            return False

        report_data["survey"] = new_draft
        return True
        

    @staticmethod
    def _derive_quality(evaluation: dict[str, dict]) -> dict[str, Any]:
        required = [
            'citation_accuracy',
            'faithfulness'
        ]
        failed: list[str] = []
        unverified: list[str] = []

        for metric in required:
            ev = evaluation.get(metric, {})
            if not ev or ev.get('skipped', False):
                unverified.append(metric)
            elif not ev.get('passed', False):
                failed.append(metric)

        if failed:
            return {"status": "failed", "failed_metrics": failed,
                    "unverified_metrics": unverified}
        if unverified:
            return {"status": "unverified", "failed_metrics": [],
                    "unverified_metrics": unverified}

        return {"status": "passed", "failed_metrics": [], "unverified_metrics": []}


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
