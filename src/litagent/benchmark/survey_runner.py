"""Run isolated, end-to-end Survey benchmark profiles."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.corpus import (
    ingest_benchmark_manifest,
    merge_manifest_bundle,
)
from litagent.benchmark.survey_judge import (
    SurveyBenchmarkJudge,
    judge_prompt_fingerprint,
)
from litagent.benchmark.survey_metrics import (
    METRICS_VERSION,
    aggregate_profiles,
    artifact_evidence_ledger,
    evaluate_survey_artifact,
    external_input_fingerprint,
)
from litagent.benchmark.survey_models import (
    ExternalInputSnapshot,
    JudgeUsage,
    SurveyBenchmarkBudget,
    SurveyBenchmarkCase,
    SurveyBenchmarkDataset,
    SurveyBenchmarkProfile,
    SurveyBenchmarkResult,
    SurveyCaseObservation,
    SurveyJudgeConfig,
    SurveyJudgeMetrics,
    SurveyProfileComparison,
    SurveyProfileSummary,
)
from litagent.benchmark.survey_recovery import (
    SurveyRecoveryStore,
    recovery_key,
    verified_product,
)
from litagent.config import AppConfig
from litagent.contracts import build_config_summary
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.rag.corpus import CollectionIdentity
from litagent.rag.manifest import load_manifest, materialize_manifest_assets
from litagent.rag.runtime import CorpusRuntime
from litagent.runner import ExternalInputReplay, LitAgent, RunPolicy

CaseExecutor = Callable[
    [SurveyBenchmarkCase, SurveyBenchmarkProfile, AppConfig, int],
    Awaitable[SurveyCaseObservation],
]

BASELINE_ELIGIBLE_JUDGMENT_STATUSES = frozenset(
    {"human_reviewed", "owner_approved_ai_assisted"}
)


def _hash_files(paths: Sequence[Path]) -> str:
    """Hash ordered file paths and bytes as one reproducibility identity."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _observation_checkpoint_key(payload: Mapping[str, Any]) -> str:
    """Bind one resumable observation to every reproducibility input."""
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_checkpoint_is_reusable(
    observation: SurveyCaseObservation,
    *,
    require_trace: bool,
) -> bool:
    """Accept only complete observations backed by an unchanged run artifact."""
    if (
        observation.status != "completed"
        or observation.metrics is None
        or observation.metrics.partial
        or observation.judge is None
        or observation.judge.status != "completed"
        or not observation.artifact_path
        or not observation.artifact_sha256
        or (
            require_trace
            and (not observation.survey_trace_id or not observation.judge_trace_id)
        )
    ):
        return False
    artifact_path = Path(observation.artifact_path)
    if not artifact_path.is_file():
        return False
    try:
        actual = f"sha256:{hashlib.sha256(artifact_path.read_bytes()).hexdigest()}"
    except OSError:
        return False
    return actual == observation.artifact_sha256


def _observation_semantically_complete(observation: SurveyCaseObservation) -> bool:
    """Require product, Judge, and delivery metrics to have terminal valid output."""
    return bool(
        observation.status == "completed"
        and observation.metrics is not None
        and not observation.metrics.partial
        and observation.judge is not None
        and observation.judge.status == "completed"
    )


def _capture_external_inputs_from_artifact(
    *,
    case: SurveyBenchmarkCase,
    repetition: int,
    artifact: Mapping[str, Any],
) -> ExternalInputSnapshot:
    """Extract the exact planner and external-search cohort from one run."""
    nodes = artifact.get("nodes")
    if not isinstance(nodes, Mapping):
        raise ValueError("run artifact has no replayable execution nodes")

    decomposition: list[str] | None = None
    search_nodes: dict[str, Mapping[str, Any]] = {}
    for node_id, node in nodes.items():
        if not isinstance(node, Mapping):
            continue
        task_id = str(node.get("task_id") or "")
        kind = str(node.get("kind") or "")
        is_subspan = kind == "subspan" or str(node_id).startswith("subspan:")
        is_worker = kind == "worker" or str(node_id).startswith("worker:")
        if is_subspan and task_id in {
            "query_decomposition",
            "planner.query_decomposition",
        }:
            output = node.get("output")
            raw_queries = (
                output.get("sub_queries") if isinstance(output, Mapping) else None
            )
            if isinstance(raw_queries, list):
                decomposition = [
                    item.strip()
                    for item in raw_queries
                    if isinstance(item, str) and item.strip()
                ]
        elif is_worker and task_id.startswith("search_"):
            search_nodes[task_id] = node

    if not decomposition:
        raise ValueError("run artifact has no planner decomposition output")
    if not search_nodes:
        raise ValueError("run artifact has no external search outputs")

    report = artifact.get("report")
    metadata = report.get("metadata") if isinstance(report, Mapping) else None
    raw_outcomes = (
        metadata.get("search_source_outcomes")
        if isinstance(metadata, Mapping)
        else None
    )
    outcomes = {
        str(item.get("task_id")): item
        for item in raw_outcomes or []
        if isinstance(item, Mapping) and item.get("task_id")
    }

    search_outputs: dict[str, dict[str, Any]] = {}
    for task_id, node in sorted(search_nodes.items()):
        input_data = node.get("input")
        output = node.get("output")
        if not isinstance(input_data, Mapping) or not isinstance(output, list):
            raise ValueError(f"search node {task_id} is not replayable")
        outcome = outcomes.get(task_id)
        if not isinstance(outcome, Mapping):
            source = str(input_data.get("source") or "")
            outcome = {
                "task_id": task_id,
                "source": source,
                "status": "success" if output else "empty",
                "result_count": len(output),
                "elapsed_ms": 0,
                "reason_code": None if output else "provider_empty",
                "error_type": None,
                "from_fallback": False,
            }
        search_outputs[task_id] = {
            "task_id": task_id,
            "input": dict(input_data),
            "output": output,
            "outcome": dict(outcome),
        }

    return ExternalInputSnapshot(
        case_id=case.case_id,
        repetition=repetition,
        query=case.query,
        decomposition=decomposition,
        search_outputs=search_outputs,
        external_input_fingerprint=external_input_fingerprint(artifact),
    )


def runtime_prompt_source_fingerprint() -> str:
    """Hash the production files that define planning and report prompts."""
    root = Path(__file__).resolve().parents[1]
    return _hash_files(
        [
            root / "agents" / "planner.py",
            root / "evidence.py",
            root / "runner.py",
            root / "eval" / "citation.py",
            root / "observability" / "citations.py",
            root / "agents" / "synthesis.py",
            root / "agents" / "reviewer.py",
            root / "agents" / "adversarial.py",
            root / "context" / "evidence_selector.py",
            root / "eval" / "ragas_eval.py",
            root / "eval" / "scoped.py",
            root / "eval" / "consistency.py",
            root / "review_structure.py",
            root / "tools" / "builtin" / "search.py",
            root / "eval" / "ragas_contract.py",
            root / "llm" / "client.py",
            root / "benchmark" / "survey_metrics.py",
            root / "rag" / "models.py",
        ]
    )


def product_source_fingerprint() -> str:
    """Fingerprint product code independently of benchmark/Judge implementation."""
    root = Path(__file__).resolve().parents[1]
    return _hash_files(
        [
            path
            for path in root.rglob("*.py")
            if "benchmark" not in path.relative_to(root).parts
        ]
    )


def _rank_key(summary: SurveyProfileSummary) -> tuple[float, ...]:
    """Rank profiles lexicographically without manufacturing a total score."""
    quality = (
        summary.valid_observation_rate,
        summary.delivery_accuracy if summary.delivery_accuracy is not None else -1.0,
        summary.coverage_evidence if summary.coverage_evidence is not None else -1.0,
        summary.citation_recall if summary.citation_recall is not None else -1.0,
        summary.faithfulness if summary.faithfulness is not None else -1.0,
        summary.topic_coverage if summary.topic_coverage is not None else -1.0,
        summary.citation_precision if summary.citation_precision is not None else -1.0,
    )
    cost = (
        -(summary.total_tokens if summary.total_tokens is not None else float("inf")),
        -(
            summary.latency_p95_ms
            if summary.latency_p95_ms is not None
            else float("inf")
        ),
    )
    return (*quality, *cost)


def _domain_summaries(observations: Sequence[SurveyCaseObservation]) -> dict[str, dict]:
    """Aggregate each domain/profile independently for diagnosis."""
    grouped: dict[tuple[str, str], list[SurveyCaseObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.domain.value, observation.profile_id)].append(observation)
    result: dict[str, dict] = {}
    for (domain, profile_id), values in sorted(grouped.items()):
        _, summaries = aggregate_profiles(values)
        result.setdefault(domain, {})[profile_id] = summaries[0].model_dump(mode="json")
    return result


def _profile_comparisons(
    observations: Sequence[SurveyCaseObservation],
    summaries: Sequence[SurveyProfileSummary],
) -> list[SurveyProfileComparison]:
    """Build pairwise deltas only where every shared external cohort matches."""
    by_profile = {summary.profile_id: summary for summary in summaries}
    by_cohort: dict[tuple[str, int], dict[str, SurveyCaseObservation]] = defaultdict(
        dict
    )
    for observation in observations:
        cohort = by_cohort[(observation.case_id, observation.repetition)]
        if observation.profile_id in cohort:
            raise ValueError("duplicate benchmark cohort observation")
        cohort[observation.profile_id] = observation
    fields = (
        "valid_observation_rate",
        "delivery_accuracy",
        "coverage_evidence",
        "citation_recall",
        "faithfulness",
        "topic_coverage",
        "citation_precision",
        "total_tokens",
        "latency_p95_ms",
    )

    def available(item: SurveyCaseObservation, field: str) -> bool:
        metrics = item.metrics
        if field == "valid_observation_rate":
            return True
        if item.status != "completed" or metrics is None:
            return False
        if field == "total_tokens":
            return metrics.cost.usage_complete
        if field == "latency_p95_ms":
            return metrics.controlled_downstream_ms is not None
        if field in {"faithfulness", "topic_coverage"}:
            return bool(item.judge and getattr(item.judge, field) is not None)
        if field == "citation_precision":
            return metrics.citations.precision is not None
        return True

    comparisons: list[SurveyProfileComparison] = []
    for left_id, right_id in combinations(sorted(by_profile), 2):
        shared = [
            values
            for values in by_cohort.values()
            if left_id in values and right_id in values
        ]
        mismatches = [
            values
            for values in shared
            if not values[left_id].external_input_fingerprint
            or values[left_id].external_input_fingerprint
            != values[right_id].external_input_fingerprint
            or any(
                getattr(values[left_id], name)
                and getattr(values[right_id], name)
                and getattr(values[left_id], name) != getattr(values[right_id], name)
                for name in (
                    "dataset_fingerprint",
                    "manifest_fingerprint",
                    "generation_model",
                    "judge_model",
                    "judge_config_fingerprint",
                    "prompt_source_fingerprint",
                    "git_sha",
                )
                if hasattr(values[left_id], name)
            )
        ]
        comparable = bool(shared) and not mismatches
        deltas: dict[str, float | None] = {}
        paired_counts: dict[str, int] = {}
        exclusions: dict[str, list[str]] = {}
        if comparable:
            for field in fields:
                exclusions[field] = [
                    f"{values[left_id].case_id}:r{values[left_id].repetition}:"
                    + ",".join(
                        f"{profile_id}_unavailable"
                        for profile_id in (left_id, right_id)
                        if not available(values[profile_id], field)
                    )
                    for values in shared
                    if not (
                        available(values[left_id], field)
                        and available(values[right_id], field)
                    )
                ]
                paired = [
                    values
                    for values in shared
                    if available(values[left_id], field)
                    and available(values[right_id], field)
                ]
                paired_counts[field] = len(paired)
                if not paired:
                    deltas[field] = None
                    continue
                paired_observations = [
                    values[profile_id]
                    for values in paired
                    for profile_id in (left_id, right_id)
                ]
                _, paired_summaries = aggregate_profiles(paired_observations)
                paired_by_profile = {
                    summary.profile_id: summary for summary in paired_summaries
                }
                summary_field = (
                    "controlled_downstream_p95_ms"
                    if field == "latency_p95_ms"
                    else field
                )
                left_value = getattr(paired_by_profile[left_id], summary_field)
                right_value = getattr(paired_by_profile[right_id], summary_field)
                deltas[field] = (
                    float(left_value) - float(right_value)
                    if left_value is not None and right_value is not None
                    else None
                )
        comparisons.append(
            SurveyProfileComparison(
                left_profile_id=left_id,
                right_profile_id=right_id,
                comparable=comparable,
                shared_cohort_count=len(shared),
                metric_paired_counts=paired_counts,
                metric_exclusion_reasons=exclusions,
                metric_deltas=deltas,
                reason_codes=([] if comparable else ["non_comparable_external_inputs"]),
            )
        )
    return comparisons


class SurveyBenchmarkRunner:
    """Execute the controlled ablation and reusable 15-case baseline."""

    def __init__(
        self,
        *,
        base_config: AppConfig,
        judge_config: SurveyJudgeConfig,
        artifacts: BenchmarkArtifactRepository,
        run_archive: ArchiveRepository,
        case_executor: CaseExecutor | None = None,
        budget: SurveyBenchmarkBudget | None = None,
    ) -> None:
        """Store configuration and optional offline execution boundary."""
        self._base_config = base_config
        self._judge_config = judge_config
        self._artifacts = artifacts
        self._run_archive = run_archive
        self._case_executor = case_executor or self._run_live_case
        self._uses_live_executor = case_executor is None
        self._judged_paper_ids: set[str] = set()
        self._budget = budget
        self._budget_observations: list[SurveyCaseObservation] = []
        self._halt_reason_codes: list[str] = []
        self._recovery = SurveyRecoveryStore(artifacts)

    def _reset_budget_state(self) -> None:
        """Reset accounting before one top-level benchmark stage."""
        self._budget_observations = []
        self._halt_reason_codes = []

    def _budget_allows_new_observation(self, *, judge_only: bool = False) -> bool:
        """Stop before the next paid run once any configured cap is reached."""
        if self._budget is None:
            return True
        product_tokens = sum(
            item.metrics.cost.total_tokens
            for item in self._budget_observations
            if item.metrics is not None
        )
        judge_tokens = sum(
            item.judge_usage.total_tokens for item in self._budget_observations
        )
        observation_count = len(self._budget_observations)
        if self._uses_live_executor:
            accounting = self._recovery.accounting()
            product_tokens = accounting["product_tokens"]
            judge_tokens = accounting["judge_tokens"]
            observation_count = accounting["products"]
            if not accounting["usage_complete"]:
                self._halt_reason_codes.append("budget_usage_unknown")
                return False
        checks = (
            (
                not judge_only and observation_count >= self._budget.max_observations,
                "budget_exhausted:max_observations",
            ),
            (
                not judge_only and product_tokens >= self._budget.max_product_tokens,
                "budget_exhausted:max_product_tokens",
            ),
            (
                judge_tokens >= self._budget.max_judge_tokens,
                "budget_exhausted:max_judge_tokens",
            ),
        )
        reasons = [reason for exhausted, reason in checks if exhausted]
        if reasons:
            self._halt_reason_codes.extend(reasons)
            self._halt_reason_codes = list(dict.fromkeys(self._halt_reason_codes))
            return False
        return True

    def _record_budget_observation(self, observation: SurveyCaseObservation) -> None:
        """Count one completed or failed observation exactly once per stage."""
        if all(item.run_id != observation.run_id for item in self._budget_observations):
            self._budget_observations.append(observation)

    def _validate_live_credentials(self) -> None:
        """Fail before paid matrix execution when either model credential is absent."""
        if not self._uses_live_executor:
            return
        if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
            raise ValueError("survey benchmark requires a generation API key")
        if not os.getenv(self._judge_config.api_key_env):
            raise ValueError(
                f"survey benchmark requires {self._judge_config.api_key_env}"
            )

    async def rejudge(
        self,
        *,
        source: Path,
        dataset: SurveyBenchmarkDataset,
        case_ids: Sequence[str] = (),
        profile_ids: Sequence[str] = (),
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Audit explicit frozen reports; never initialize product or corpus services."""
        raw = source.read_bytes()
        original = SurveyBenchmarkResult.model_validate_json(raw)
        selected = [
            item
            for item in original.observations
            if (not case_ids or item.case_id in case_ids)
            and (not profile_ids or item.profile_id in profile_ids)
        ]
        if (
            not selected
            or set(case_ids) - {item.case_id for item in selected}
            or set(profile_ids) - {item.profile_id for item in selected}
        ):
            raise ValueError("judge_only_selection_not_found")
        entries = []
        for item in selected:
            artifact = verified_product(item)
            case = dataset.case_by_id(item.case_id)
            if artifact.get("query") != case.query:
                raise ValueError("judge_only_query_mismatch")
            entries.append((item, artifact, case))
        actions = []
        for item, artifact, case in entries:
            key = self._judge_key(case, item, artifact)
            state = self._recovery.judge(key)
            actions.append(
                {
                    "run_id": item.run_id,
                    "product": "reuse",
                    "judge": (
                        "reuse"
                        if state.get("metrics", {}).get("status") == "completed"
                        else (
                            "audit" if state.get("inventory") else "inventory_and_audit"
                        )
                    ),
                    "artifact_sha256": item.artifact_sha256,
                }
            )
            if (
                item.status != "completed"
                or item.metrics is None
                or item.metrics.partial
            ):
                actions[-1]["judge"] = "blocked_product_incomplete"
        payload = {
            "run_id": f"survey-rejudge-{uuid.uuid4().hex}",
            "benchmark_type": "survey_rejudge",
            "status": "planned",
            "formal_eligible": False,
            "source_path": str(source.resolve()),
            "source_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
            "judge_config_fingerprint": self._judge_config.fingerprint,
            "judge_prompt_fingerprint": judge_prompt_fingerprint(),
            "actions": actions,
            "observations": [],
            "reason_codes": ["diagnostic_rejudge_not_baseline"],
        }
        if dry_run:
            return payload
        if self._budget is None:
            raise ValueError("judge-only execution requires explicit budgets")
        if not os.getenv(self._judge_config.api_key_env):
            raise ValueError(f"judge-only requires {self._judge_config.api_key_env}")
        if len(entries) > self._budget.max_observations:
            raise ValueError("judge-only selection exceeds max_observations")
        self._reset_budget_state()
        # Import source provenance separately; these bytes are never rewritten.
        for item, artifact, case in entries:
            slot = recovery_key({"imported_artifact": item.artifact_sha256})
            if self._recovery.product(slot, item.artifact_sha256) is None:
                self._recovery.save_product(slot, item.artifact_sha256, item)
            result = await self._judge_product(case, self._base_config, item, artifact)
            payload["observations"].append(result.model_dump(mode="json"))
            payload["status"] = "partial"
            payload["accounting"] = self._recovery.accounting()
            self._artifacts.write(payload)
            if result.judge is None or result.judge.status != "completed":
                break
        if len(payload["observations"]) == len(entries) and all(
            item["judge"] and item["judge"]["status"] == "completed"
            for item in payload["observations"]
        ):
            payload["status"] = "succeeded"
        self._artifacts.write(payload)
        return payload

    def _judge_key(
        self,
        case: SurveyBenchmarkCase,
        product: SurveyCaseObservation,
        artifact: dict[str, Any],
    ) -> str:
        """Bind audit recovery to exact inputs and the current Judge contract."""
        return recovery_key(
            {
                "survey": str((artifact.get("report") or {}).get("survey") or ""),
                "ledger": artifact_evidence_ledger(artifact),
                "artifact_sha": product.artifact_sha256,
                "case": case.model_dump(mode="json"),
                "config": self._judge_config.fingerprint,
                "prompt": judge_prompt_fingerprint(),
            }
        )

    def plan_matrix(
        self,
        dataset: SurveyBenchmarkDataset,
        cases: Sequence[SurveyBenchmarkCase],
        profiles: Sequence[SurveyBenchmarkProfile],
    ) -> dict[str, Any]:
        """Inspect first-pass recovery actions without credentials or service access."""
        actions = []
        for case in cases:
            for profile in profiles:
                slot, identity = self._product_identity(dataset, case, profile, 1)
                try:
                    product = self._recovery.product(slot, identity)
                    action = {
                        "case_id": case.case_id,
                        "profile_id": profile.profile_id,
                        "product": "reuse" if product else "generate",
                    }
                    if product:
                        state = self._recovery.judge(
                            self._judge_key(case, product, verified_product(product))
                        )
                        action["judge"] = (
                            "reuse"
                            if state.get("metrics", {}).get("status") == "completed"
                            else (
                                "audit"
                                if state.get("inventory")
                                else "inventory_and_audit"
                            )
                        )
                    else:
                        action["judge"] = "inventory_and_audit"
                    if product and (
                        product.status != "completed"
                        or product.metrics is None
                        or product.metrics.partial
                    ):
                        action["judge"] = "blocked_product_incomplete"
                except (ValueError, OSError) as exc:
                    action = {
                        "case_id": case.case_id,
                        "profile_id": profile.profile_id,
                        "product": "blocked",
                        "reason": str(exc),
                    }
                actions.append(action)
        return {
            "status": "planned",
            "actions": actions,
            "accounting": self._recovery.accounting(),
            "budget": self._budget.model_dump() if self._budget else None,
        }

    def _product_identity(
        self,
        dataset: SurveyBenchmarkDataset,
        case: SurveyBenchmarkCase,
        profile: SurveyBenchmarkProfile,
        repetition: int,
    ) -> tuple[str, str]:
        """Keep a stable slot and reject changed product inputs within an output root."""
        slot = recovery_key(
            {
                "case_id": case.case_id,
                "profile_id": profile.profile_id,
                "repetition": repetition,
            }
        )
        identity = recovery_key(
            {
                "case": case.model_dump(mode="json"),
                "profile": profile.fingerprint,
                "config": build_config_summary(
                    self._config_for(profile, dataset.dataset_version)
                )["fingerprint"],
                "product_source": product_source_fingerprint(),
                "dataset": dataset.fingerprint,
                "manifest": _hash_files(
                    [Path(path) for path in dataset.manifest_paths]
                ),
            }
        )
        return slot, identity

    def _config_for(
        self, profile: SurveyBenchmarkProfile, dataset_version: str
    ) -> AppConfig:
        """Build one isolated effective application configuration."""
        planner = self._base_config.planner.model_copy(update={"temperature": 0.0})
        if not profile.rag_enabled:
            rag = self._base_config.rag.model_copy(update={"enabled": False})
        else:
            assert profile.rag_profile is not None
            rag = profile.rag_profile.apply(self._base_config.rag).model_copy(
                update={
                    "enabled": True,
                    "paper_collection": "papers_survey_benchmark",
                    "corpus_version": f"survey-{dataset_version}",
                    "writeback_enabled": False,
                }
            )
        return self._base_config.model_copy(
            update={"planner": planner, "rag": rag},
            deep=True,
        )

    @staticmethod
    def _finalize_survey_artifact(
        recorder: RunRecorder,
        tracer: LangFuseTracer,
        *,
        report: Mapping[str, Any] | None = None,
        error: str | None = None,
        terminal_status: str,
    ) -> dict[str, Any]:
        """Persist a run only after attaching its remote trace identity."""
        recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
        return recorder.finalize(
            report,
            error=error,
            terminal_status=terminal_status,
        )

    async def _prepare_corpora(
        self,
        dataset: SurveyBenchmarkDataset,
        profiles: Sequence[SurveyBenchmarkProfile],
    ) -> None:
        """Build every enabled profile collection from the same paper universe."""
        enabled = [profile for profile in profiles if profile.rag_enabled]
        if not enabled:
            return
        config = self._config_for(enabled[0], dataset.dataset_version)
        manifest_path = merge_manifest_bundle(
            [Path(path) for path in dataset.manifest_paths],
            raw_root=Path(config.rag.raw_root),
            corpus_version=config.rag.corpus_version,
            output_path=self._artifacts.root / "corpus" / "merged_manifest.yaml",
            include_paper_ids=set(dataset.corpus_paper_ids),
        )
        manifest = load_manifest(manifest_path, raw_root=Path(config.rag.raw_root))
        manifest_ids = {
            asset.paper_id for asset in materialize_manifest_assets(manifest)
        }
        if manifest_ids != set(dataset.corpus_paper_ids):
            raise ValueError("survey corpus_paper_ids do not match manifest bundle")

        prepared: set[str] = set()
        for profile in enabled:
            effective = self._config_for(profile, dataset.dataset_version)
            identity = CollectionIdentity.from_config(effective.rag, purpose="runtime")
            if identity.fingerprint in prepared:
                continue
            runtime = await CorpusRuntime.connect(effective, purpose="runtime")
            try:
                await ingest_benchmark_manifest(
                    config=effective,
                    runtime=runtime,
                    manifest_path=manifest_path,
                )
                stats = await runtime.store.stats()
                if stats.points_count <= 0:
                    raise RuntimeError("empty_survey_benchmark_index")
            finally:
                await runtime.close()
            prepared.add(identity.fingerprint)

    async def _run_matrix(
        self,
        *,
        dataset: SurveyBenchmarkDataset,
        cases: Sequence[SurveyBenchmarkCase],
        profiles: Sequence[SurveyBenchmarkProfile],
        repetition: int,
        git_sha: str,
        git_dirty: bool,
    ) -> list[SurveyCaseObservation]:
        """Run cases round-robin so external API drift affects profiles equally."""
        observations: list[SurveyCaseObservation] = []
        prompt_fingerprint = runtime_prompt_source_fingerprint()
        judge_fingerprint = judge_prompt_fingerprint()
        dataset_fingerprint = dataset.fingerprint
        manifest_fingerprint = _hash_files(
            [Path(path) for path in dataset.manifest_paths]
        )
        for case in cases:
            snapshot: ExternalInputSnapshot | None = None
            snapshot_key = _observation_checkpoint_key(
                {
                    "case_id": case.case_id,
                    "repetition": repetition,
                    "generation_model": self._base_config.llm.model,
                    "planner": self._base_config.planner.model_dump(mode="json"),
                    "semantic_scholar_enabled": bool(
                        os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
                    ),
                    "product_source_fingerprint": product_source_fingerprint(),
                    "dataset_fingerprint": dataset_fingerprint,
                    "manifest_fingerprint": manifest_fingerprint,
                }
            )
            if self._uses_live_executor:
                raw_snapshot = self._artifacts.read_checkpoint(
                    "survey-external-inputs", snapshot_key
                )
                if raw_snapshot is not None:
                    try:
                        candidate_snapshot = ExternalInputSnapshot.model_validate(
                            raw_snapshot
                        )
                    except ValueError:
                        candidate_snapshot = None
                    if (
                        candidate_snapshot is not None
                        and candidate_snapshot.case_id == case.case_id
                        and candidate_snapshot.repetition == repetition
                        and candidate_snapshot.query == case.query
                    ):
                        snapshot = candidate_snapshot

            for profile_index, profile in enumerate(profiles):
                config = self._config_for(profile, dataset.dataset_version)
                config_fingerprint = build_config_summary(config)["fingerprint"]
                checkpoint_key = _observation_checkpoint_key(
                    {
                        "case_id": case.case_id,
                        "profile_id": profile.profile_id,
                        "profile_fingerprint": profile.fingerprint,
                        "repetition": repetition,
                        "generation_model": self._base_config.llm.model,
                        "judge_model": self._judge_config.model,
                        "judge_config_fingerprint": self._judge_config.fingerprint,
                        "git_sha": git_sha,
                        "git_dirty": git_dirty,
                        "config_fingerprint": config_fingerprint,
                        "prompt_source_fingerprint": prompt_fingerprint,
                        "judge_prompt_fingerprint": judge_fingerprint,
                        "dataset_fingerprint": dataset_fingerprint,
                        "manifest_fingerprint": manifest_fingerprint,
                    }
                )
                checkpoint = self._artifacts.read_checkpoint(
                    "survey-observations",
                    checkpoint_key,
                    strict=self._uses_live_executor,
                )
                observation = None
                product_slot, product_identity = self._product_identity(
                    dataset, case, profile, repetition
                )
                product = (
                    self._recovery.product(product_slot, product_identity)
                    if self._uses_live_executor
                    else None
                )
                if checkpoint is not None:
                    try:
                        candidate = SurveyCaseObservation.model_validate(checkpoint)
                    except ValueError:
                        candidate = None
                    if (
                        candidate is not None
                        and _observation_checkpoint_is_reusable(
                            candidate,
                            require_trace=config.observability.enabled,
                        )
                        and (
                            snapshot is None
                            or candidate.external_input_fingerprint
                            == snapshot.external_input_fingerprint
                        )
                    ):
                        observation = candidate
                    elif self._uses_live_executor and product is None:
                        raise ValueError(
                            "legacy_observation_requires_explicit_judge_only_import"
                        )

                if observation is None and product is not None:
                    artifact = verified_product(product)
                    product = product.model_copy(
                        update={
                            "metrics": evaluate_survey_artifact(
                                case=case,
                                artifact=artifact,
                                judged_paper_ids=self._judged_paper_ids
                                or set(dataset.corpus_paper_ids),
                            )
                        }
                    )
                    if (
                        snapshot is not None
                        and product.external_input_fingerprint
                        != snapshot.external_input_fingerprint
                    ):
                        raise ValueError("product_external_inputs_mismatch")
                    observation = await self._judge_product(
                        case, config, product, artifact
                    )

                if observation is None:
                    if not self._budget_allows_new_observation():
                        return observations
                    if (
                        self._uses_live_executor
                        and profile_index > 0
                        and snapshot is None
                    ):
                        self._halt_reason_codes.append(
                            f"external_input_snapshot_unavailable:{case.case_id}:"
                            f"r{repetition}"
                        )
                        break
                    try:
                        if self._uses_live_executor:
                            replay = (
                                ExternalInputReplay(
                                    query=snapshot.query,
                                    decomposition=tuple(snapshot.decomposition),
                                    search_outputs=snapshot.search_outputs,
                                    fingerprint=snapshot.fingerprint,
                                )
                                if snapshot is not None
                                else None
                            )
                            observation = await self._run_live_case(
                                case,
                                profile,
                                config,
                                repetition,
                                external_inputs=replay,
                                product_slot=product_slot,
                                product_identity=product_identity,
                                provenance={
                                    "git_sha": git_sha,
                                    "git_dirty": git_dirty,
                                    "profile_fingerprint": profile.fingerprint,
                                    "prompt_source_fingerprint": product_source_fingerprint(),
                                    "dataset_fingerprint": dataset_fingerprint,
                                    "manifest_fingerprint": manifest_fingerprint,
                                },
                            )
                        else:
                            observation = await self._case_executor(
                                case, profile, config, repetition
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        observation = SurveyCaseObservation(
                            run_id=(
                                f"failed-{profile.profile_id}-{case.case_id}-"
                                f"r{repetition}-{uuid.uuid4().hex[:8]}"
                            ),
                            case_id=case.case_id,
                            domain=case.domain,
                            profile_id=profile.profile_id,
                            repetition=repetition,
                            status="failed",
                            reason_codes=[f"case_executor_failed:{type(exc).__name__}"],
                        )
                observation = observation.model_copy(
                    update={
                        "generation_model": self._base_config.llm.model,
                        "judge_model": self._judge_config.model,
                        "git_sha": observation.git_sha or git_sha,
                        "git_dirty": (
                            observation.git_dirty
                            if observation.git_dirty is not None
                            else git_dirty
                        ),
                        "config_fingerprint": config_fingerprint,
                        "profile_fingerprint": profile.fingerprint,
                        "prompt_source_fingerprint": prompt_fingerprint,
                        "dataset_fingerprint": dataset_fingerprint,
                        "manifest_fingerprint": manifest_fingerprint,
                    }
                )
                self._artifacts.write_checkpoint(
                    "survey-observations", checkpoint_key, observation
                )
                observations.append(observation)
                self._record_budget_observation(observation)

                if self._uses_live_executor and snapshot is None:
                    artifact = self._run_archive.get(observation.run_id)
                    if artifact is not None:
                        try:
                            snapshot = _capture_external_inputs_from_artifact(
                                case=case,
                                repetition=repetition,
                                artifact=artifact,
                            )
                        except ValueError:
                            self._halt_reason_codes.append(
                                f"external_input_snapshot_unavailable:{case.case_id}:"
                                f"r{repetition}"
                            )
                        else:
                            self._artifacts.write_checkpoint(
                                "survey-external-inputs", snapshot_key, snapshot
                            )
                if not _observation_semantically_complete(observation):
                    self._halt_reason_codes.append(
                        f"observation_terminal_gate:{case.case_id}:"
                        f"{profile.profile_id}:r{repetition}"
                    )
                    return observations
        return observations

    @staticmethod
    def _comparable_observations(
        observations: Sequence[SurveyCaseObservation],
        profile_ids: set[str],
    ) -> tuple[list[SurveyCaseObservation], list[str]]:
        """Keep only case/repetition cohorts with identical external inputs."""
        grouped: dict[tuple[str, int], list[SurveyCaseObservation]] = defaultdict(list)
        for observation in observations:
            grouped[(observation.case_id, observation.repetition)].append(observation)
        comparable: list[SurveyCaseObservation] = []
        reasons: list[str] = []
        for (case_id, repetition), values in sorted(grouped.items()):
            value_profiles = {item.profile_id for item in values}
            fingerprints = {item.external_input_fingerprint for item in values}
            if (
                value_profiles == profile_ids
                and len(fingerprints) == 1
                and None not in fingerprints
            ):
                comparable.extend(values)
            else:
                reasons.append(
                    f"non_comparable_external_inputs:{case_id}:r{repetition}"
                )
        return comparable, reasons

    async def run_pilot(
        self,
        *,
        dataset: SurveyBenchmarkDataset,
        cases: Sequence[SurveyBenchmarkCase],
        profiles: Sequence[SurveyBenchmarkProfile],
        git_sha: str,
        git_dirty: bool = False,
    ) -> SurveyBenchmarkResult:
        """Run one explicit cost-controlled matrix without selecting a winner."""
        if not cases:
            raise ValueError("survey pilot requires at least one case")
        if not profiles:
            raise ValueError("survey pilot requires at least one profile")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("survey pilot case ids must be unique")
        if len({profile.profile_id for profile in profiles}) != len(profiles):
            raise ValueError("survey pilot profile ids must be unique")

        self._reset_budget_state()
        self._validate_live_credentials()
        if self._uses_live_executor:
            await self._prepare_corpora(dataset, profiles)
        self._judged_paper_ids = set(dataset.corpus_paper_ids)

        observations = await self._run_matrix(
            dataset=dataset,
            cases=cases,
            profiles=profiles,
            repetition=1,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        comparable, reasons = self._comparable_observations(
            observations, {profile.profile_id for profile in profiles}
        )
        reasons.extend(self._halt_reason_codes)
        if len(comparable) != len(cases) * len(profiles):
            reasons.append("pilot_matrix_incomplete")

        case_aggregates, summaries = aggregate_profiles(observations)
        result = self._build_result(
            stage="pilot",
            dataset=dataset,
            profiles=profiles,
            observations=observations,
            case_aggregates=case_aggregates,
            summaries=summaries,
            recommended_profile_id=None,
            git_sha=git_sha,
            git_dirty=git_dirty,
            reason_codes=reasons,
        )
        self._artifacts.write(result)
        return result

    async def run_ablation(
        self,
        *,
        dataset: SurveyBenchmarkDataset,
        profiles: Sequence[SurveyBenchmarkProfile],
        git_sha: str,
        git_dirty: bool = False,
    ) -> SurveyBenchmarkResult:
        """Run 18 first-pass and 12 second-pass controlled observations."""
        if len(profiles) != 3:
            raise ValueError("survey ablation requires exactly three profiles")
        if len({profile.profile_id for profile in profiles}) != 3:
            raise ValueError("survey profile ids must be unique")
        self._reset_budget_state()
        self._validate_live_credentials()
        if self._uses_live_executor:
            await self._prepare_corpora(dataset, profiles)
        self._judged_paper_ids = set(dataset.corpus_paper_ids)
        cases = [dataset.case_by_id(case_id) for case_id in dataset.ablation_case_ids]
        first_pass = await self._run_matrix(
            dataset=dataset,
            cases=cases,
            profiles=profiles,
            repetition=1,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        comparable_first, reasons = self._comparable_observations(
            first_pass, {profile.profile_id for profile in profiles}
        )
        reasons.extend(self._halt_reason_codes)
        first_pass_incomplete = any(
            not _observation_semantically_complete(item) for item in first_pass
        )
        if first_pass_incomplete:
            reasons.append("first_pass_incomplete")
        if not comparable_first or self._halt_reason_codes or first_pass_incomplete:
            case_aggregates, summaries = aggregate_profiles(first_pass)
            result = self._build_result(
                stage="ablation",
                dataset=dataset,
                profiles=profiles,
                observations=first_pass,
                case_aggregates=case_aggregates,
                summaries=summaries,
                recommended_profile_id=None,
                git_sha=git_sha,
                git_dirty=git_dirty,
                reason_codes=reasons,
            )
            self._artifacts.write(result)
            return result

        _, first_summaries = aggregate_profiles(comparable_first)
        top_ids = [
            summary.profile_id
            for summary in sorted(first_summaries, key=_rank_key, reverse=True)[:2]
        ]
        top_profiles = [
            profile for profile in profiles if profile.profile_id in top_ids
        ]
        second_pass = await self._run_matrix(
            dataset=dataset,
            cases=cases,
            profiles=top_profiles,
            repetition=2,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        observations = [*first_pass, *second_pass]
        comparable_second, second_reasons = self._comparable_observations(
            second_pass, set(top_ids)
        )
        reasons.extend(second_reasons)
        reasons.extend(self._halt_reason_codes)
        case_aggregates, summaries = aggregate_profiles(observations)
        eligible_summaries = [
            summary for summary in summaries if summary.profile_id in set(top_ids)
        ]
        recommended = (
            max(eligible_summaries, key=_rank_key).profile_id
            if eligible_summaries and comparable_second and not self._halt_reason_codes
            else None
        )
        result = self._build_result(
            stage="ablation",
            dataset=dataset,
            profiles=profiles,
            observations=observations,
            case_aggregates=case_aggregates,
            summaries=summaries,
            recommended_profile_id=recommended,
            git_sha=git_sha,
            git_dirty=git_dirty,
            reason_codes=reasons,
        )
        self._artifacts.write(result)
        return result

    async def run_baseline(
        self,
        *,
        dataset: SurveyBenchmarkDataset,
        profiles: Sequence[SurveyBenchmarkProfile],
        ablation: SurveyBenchmarkResult,
        git_sha: str,
        git_dirty: bool = False,
    ) -> SurveyBenchmarkResult:
        """Reuse the winner's ablation runs and execute the remaining nine cases."""
        if dataset.judgment_status not in BASELINE_ELIGIBLE_JUDGMENT_STATUSES:
            raise ValueError(
                "formal baseline requires a human-reviewed or "
                "owner-approved AI-assisted dataset"
            )
        self._reset_budget_state()
        if ablation.stage != "ablation" or not ablation.recommended_profile_id:
            raise ValueError("baseline requires a completed ablation recommendation")
        config_fingerprint = build_config_summary(self._base_config)["fingerprint"]
        manifest_fingerprint = _hash_files(
            [Path(path) for path in dataset.manifest_paths]
        )
        current_profile_fingerprints = {
            profile.profile_id: profile.fingerprint for profile in profiles
        }
        if (
            ablation.dataset_fingerprint != dataset.fingerprint
            or ablation.git_sha != git_sha
            or ablation.git_dirty != git_dirty
            or ablation.config_fingerprint != config_fingerprint
            or ablation.manifest_fingerprint != manifest_fingerprint
            or ablation.prompt_source_fingerprint != runtime_prompt_source_fingerprint()
            or ablation.judge_prompt_fingerprint != judge_prompt_fingerprint()
            or ablation.generation_model != self._base_config.llm.model
            or ablation.judge_model != self._judge_config.model
            or ablation.judge_config_fingerprint != self._judge_config.fingerprint
        ):
            raise ValueError("ablation artifact is incompatible with baseline run")
        profile = next(
            (
                item
                for item in profiles
                if item.profile_id == ablation.recommended_profile_id
            ),
            None,
        )
        if profile is None:
            raise ValueError("recommended profile is absent from profile set")
        if ablation.profile_fingerprints.get(profile.profile_id) != (
            current_profile_fingerprints[profile.profile_id]
        ):
            raise ValueError("recommended profile changed since ablation")
        self._validate_live_credentials()
        if self._uses_live_executor:
            await self._prepare_corpora(dataset, [profile])
        self._judged_paper_ids = set(dataset.corpus_paper_ids)
        reused = [
            item
            for item in ablation.observations
            if item.profile_id == profile.profile_id
        ]
        remaining = [
            case
            for case in dataset.cases
            if case.case_id not in set(dataset.ablation_case_ids)
        ]
        new_observations = await self._run_matrix(
            dataset=dataset,
            cases=remaining,
            profiles=[profile],
            repetition=1,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        observations = [*reused, *new_observations]
        case_aggregates, summaries = aggregate_profiles(observations)
        result = self._build_result(
            stage="baseline",
            dataset=dataset,
            profiles=[profile],
            observations=observations,
            case_aggregates=case_aggregates,
            summaries=summaries,
            recommended_profile_id=(
                profile.profile_id if not self._halt_reason_codes else None
            ),
            git_sha=git_sha,
            git_dirty=git_dirty,
            reason_codes=list(self._halt_reason_codes),
            reused_ablation_run_id=ablation.run_id,
        )
        self._artifacts.write(result)
        return result

    def _build_result(
        self,
        *,
        stage: str,
        dataset: SurveyBenchmarkDataset,
        profiles: Sequence[SurveyBenchmarkProfile],
        observations: list[SurveyCaseObservation],
        case_aggregates,
        summaries,
        recommended_profile_id: str | None,
        git_sha: str,
        git_dirty: bool,
        reason_codes: list[str],
        reused_ablation_run_id: str | None = None,
    ) -> SurveyBenchmarkResult:
        """Assemble one reproducible terminal benchmark artifact."""
        self_judged = self._judge_config.model == self._base_config.llm.model
        final_reasons = list(reason_codes)
        incomplete = False
        for item in observations:
            identity = f"{item.case_id}:{item.profile_id}:r{item.repetition}"
            if item.status != "completed":
                incomplete = True
                final_reasons.append(f"failed_observation:{identity}")
            if item.metrics is None:
                incomplete = True
                final_reasons.append(f"missing_metrics:{identity}")
            elif item.metrics.partial:
                incomplete = True
                final_reasons.append(f"partial_observation:{identity}")
            if item.judge is None or item.judge.status != "completed":
                incomplete = True
                final_reasons.append(f"incomplete_judge:{identity}")
            if bool(item.artifact_path) != bool(item.artifact_sha256):
                incomplete = True
                final_reasons.append(f"artifact_validation_failed:{identity}")
            elif item.artifact_path and item.artifact_sha256:
                path = Path(item.artifact_path)
                try:
                    actual = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
                except OSError:
                    actual = ""
                if actual != item.artifact_sha256:
                    incomplete = True
                    final_reasons.append(f"artifact_validation_failed:{identity}")

        profile_comparisons = _profile_comparisons(observations, summaries)
        comparable_pairs = [
            comparison for comparison in profile_comparisons if comparison.comparable
        ]
        if stage == "ablation" and len(profiles) > 1 and not comparable_pairs:
            final_reasons.append("no_comparable_profile_cohort")
            recommended_profile_id = None
        if incomplete:
            recommended_profile_id = None
            final_reasons.append("incomplete_observations")

        status = "partial" if incomplete or final_reasons else "succeeded"
        formal_eligible = (
            status == "succeeded"
            and stage != "pilot"
            and dataset.judgment_status in BASELINE_ELIGIBLE_JUDGMENT_STATUSES
        )
        if self_judged:
            final_reasons.append("self_judged")
        if dataset.judgment_status not in BASELINE_ELIGIBLE_JUDGMENT_STATUSES:
            final_reasons.append("dataset_judgment_not_approved")
        elif dataset.judgment_status == "owner_approved_ai_assisted":
            final_reasons.append("owner_approved_ai_assisted_ground_truth")
        manifest_paths = [Path(path) for path in dataset.manifest_paths]
        config_summary = build_config_summary(self._base_config)
        return SurveyBenchmarkResult(
            metrics_version=METRICS_VERSION,
            run_id=f"survey-{stage}-{uuid.uuid4().hex}",
            status=status,
            stage=stage,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            dataset_fingerprint=dataset.fingerprint,
            judgment_status=dataset.judgment_status,
            annotation_version=dataset.annotation_version,
            manifest_fingerprint=_hash_files(manifest_paths),
            git_sha=git_sha,
            git_dirty=git_dirty,
            generation_model=self._base_config.llm.model,
            judge_model=self._judge_config.model,
            judge_config_fingerprint=self._judge_config.fingerprint,
            self_judged=self_judged,
            formal_eligible=formal_eligible,
            prompt_source_fingerprint=runtime_prompt_source_fingerprint(),
            judge_prompt_fingerprint=judge_prompt_fingerprint(),
            config_fingerprint=config_summary["fingerprint"],
            profile_fingerprints={
                profile.profile_id: profile.fingerprint for profile in profiles
            },
            observations=observations,
            case_aggregates=case_aggregates,
            profile_summaries=summaries,
            domain_summaries=_domain_summaries(observations),
            recommended_profile_id=recommended_profile_id,
            reused_ablation_run_id=reused_ablation_run_id,
            comparable_profile_pairs=[
                [comparison.left_profile_id, comparison.right_profile_id]
                for comparison in comparable_pairs
            ],
            profile_comparisons=profile_comparisons,
            reason_codes=list(dict.fromkeys(final_reasons)),
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        )

    async def _run_live_case(
        self,
        case: SurveyBenchmarkCase,
        profile: SurveyBenchmarkProfile,
        config: AppConfig,
        repetition: int,
        *,
        external_inputs: ExternalInputReplay | None = None,
        product_slot: str,
        product_identity: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> SurveyCaseObservation:
        """Run one real Survey, persist it, then invoke the independent Judge."""
        self._recovery.save_product(product_slot, product_identity, None)
        run_id = (
            f"survey-{profile.profile_id}-{case.case_id}-r{repetition}-"
            f"{uuid.uuid4().hex[:12]}"
        )
        recorder = RunRecorder(run_id, case.query, self._run_archive)
        recorder.set_config_summary(build_config_summary(config))
        survey_tracer = LangFuseTracer(
            host=config.observability.langfuse_host,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            trace_seed=run_id,
        )
        hooks: list[Any] = [recorder]
        if config.observability.enabled:
            hooks.append(
                RedactingTraceHook(
                    survey_tracer,
                    payload_mode=config.observability.payload_mode,
                )
            )
        trace_hook = CompositeTraceHook(*hooks)
        try:
            async with LitAgent(
                config,
                trace_hook=trace_hook,
                run_policy=RunPolicy.isolated_benchmark(),
                external_inputs=external_inputs,
            ) as agent:
                report = await agent.run(case.query)
            artifact = self._finalize_survey_artifact(
                recorder,
                survey_tracer,
                report=report,
                terminal_status="completed",
            )
        except BaseException as exc:
            error = f"survey_run_failed:{type(exc).__name__}"
            self._finalize_survey_artifact(
                recorder,
                survey_tracer,
                error=error,
                terminal_status="failed",
            )
            artifact_path = self._run_archive.path_for(run_id)
            failed_product = SurveyCaseObservation(
                run_id=run_id,
                case_id=case.case_id,
                domain=case.domain,
                profile_id=profile.profile_id,
                repetition=repetition,
                status="failed",
                artifact_id=run_id,
                artifact_path=str(artifact_path),
                artifact_sha256=(
                    f"sha256:{hashlib.sha256(artifact_path.read_bytes()).hexdigest()}"
                ),
                survey_trace_id=survey_tracer.trace_id,
                survey_trace_url=survey_tracer.trace_url,
                reason_codes=[error],
            )
            self._recovery.save_product(product_slot, product_identity, failed_product)
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            return failed_product

        artifact_path = self._run_archive.path_for(run_id)
        metrics = evaluate_survey_artifact(
            case=case,
            artifact=artifact,
            judged_paper_ids=self._judged_paper_ids,
        )
        product = SurveyCaseObservation(
            run_id=run_id,
            case_id=case.case_id,
            domain=case.domain,
            profile_id=profile.profile_id,
            repetition=repetition,
            status="completed",
            metrics=metrics,
            artifact_id=run_id,
            artifact_path=str(artifact_path),
            artifact_sha256=f"sha256:{hashlib.sha256(artifact_path.read_bytes()).hexdigest()}",
            survey_trace_id=survey_tracer.trace_id,
            survey_trace_url=survey_tracer.trace_url,
            external_input_fingerprint=external_input_fingerprint(artifact),
            generation_model=config.llm.model,
            config_fingerprint=build_config_summary(config)["fingerprint"],
        ).model_copy(update=dict(provenance or {}))
        self._recovery.save_product(product_slot, product_identity, product)
        return await self._judge_product(case, config, product, artifact)

    async def _judge_product(
        self,
        case: SurveyBenchmarkCase,
        config: AppConfig,
        product: SurveyCaseObservation,
        artifact: dict[str, Any],
    ) -> SurveyCaseObservation:
        """Audit a frozen product, persisting inventory and all paid attempts."""
        if (
            product.status != "completed"
            or product.metrics is None
            or product.metrics.partial
        ):
            return product.model_copy(
                update={
                    "judge": SurveyJudgeMetrics(
                        status="skipped", reason_codes=["product_incomplete"]
                    ),
                    "judge_usage": JudgeUsage(usage_complete=True),
                    "reason_codes": ["product_incomplete"],
                }
            )
        survey = str((artifact.get("report") or {}).get("survey") or "")
        ledger = artifact_evidence_ledger(artifact)
        key = self._judge_key(case, product, artifact)
        state = self._recovery.judge(key)
        if state.get("metrics", {}).get("status") == "completed":
            cached_judge = SurveyJudgeMetrics.model_validate(state["metrics"])
            return product.model_copy(
                update={
                    "judge": cached_judge,
                    "judge_usage": JudgeUsage.model_validate(state["usage"]),
                    "judge_trace_id": state.get("trace_id"),
                    "judge_trace_url": state.get("trace_url"),
                    "judge_model": self._judge_config.model,
                    "reason_codes": list(cached_judge.reason_codes),
                }
            )
        if not self._budget_allows_new_observation(judge_only=True):
            return product.model_copy(
                update={
                    "judge": SurveyJudgeMetrics(
                        status="skipped", reason_codes=["judge_budget_exhausted"]
                    )
                }
            )
        run_id = product.run_id
        judge_trace = LangFuseTracer(
            host=config.observability.langfuse_host,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            trace_seed=f"{run_id}:judge:{uuid.uuid4().hex}",
        )
        judge_hook = RedactingTraceHook(
            judge_trace,
            payload_mode=config.observability.payload_mode,
        )
        judge_hook("survey.start", {"query": case.query, "session_id": run_id})
        previous_tokens = sum(
            call.get("prompt_tokens", 0) + call.get("completion_tokens", 0)
            for call in state.get("calls", [])
        )
        judge_client = OpenAICompatibleClient(
            base_url=self._judge_config.base_url,
            model=self._judge_config.model,
            max_tokens=self._judge_config.max_tokens,
            temperature=0,
            trace_hook=judge_hook,
            api_key=os.getenv(self._judge_config.api_key_env, ""),
            transport_max_attempts=1,
        )
        state["trace_id"] = judge_trace.trace_id
        state["trace_url"] = judge_trace.trace_url
        state.setdefault("trace_history", []).append(judge_trace.trace_id)
        try:
            judge, judge_usage = await SurveyBenchmarkJudge(
                judge_client, self._judge_config, trace_hook=judge_hook
            ).evaluate(
                case=case,
                survey=survey,
                ledger=ledger,
                checkpoint=state,
                save_checkpoint=lambda value: self._recovery.save_judge(key, value),
                before_call=lambda: self._budget_allows_new_observation(
                    judge_only=True
                ),
            )
            state.update(
                metrics=judge.model_dump(mode="json"),
                usage=judge_usage.model_dump(mode="json"),
            )
            self._recovery.save_judge(key, state)
        except BaseException as exc:
            judge_hook(
                "survey.failed",
                {
                    "error": type(exc).__name__,
                    "reason_code": (
                        "judge_cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "judge_execution_failed"
                    ),
                },
            )
            raise
        finally:
            try:
                await judge_client.close()
            finally:
                judge_hook.flush()
        judge_hook(
            "survey.complete",
            {
                "quality_status": (
                    "passed" if judge.status == "completed" else "unverified"
                ),
                "total_tokens": judge_usage.total_tokens - previous_tokens,
                "cumulative_judge_tokens": judge_usage.total_tokens,
                "delivery_status": (
                    product.metrics.delivery_status if product.metrics else "unknown"
                ),
                "judge": judge.model_dump(mode="json"),
            },
        )
        judge_hook.flush()
        return product.model_copy(
            update={
                "judge": judge,
                "judge_usage": judge_usage,
                "judge_trace_id": judge_trace.trace_id,
                "judge_trace_url": judge_trace.trace_url,
                "judge_model": self._judge_config.model,
                "reason_codes": list(judge.reason_codes),
            }
        )
