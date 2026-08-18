"""Run isolated, end-to-end Survey benchmark profiles."""

from __future__ import annotations

import asyncio
import hashlib
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
    aggregate_profiles,
    artifact_evidence_ledger,
    evaluate_survey_artifact,
    external_input_fingerprint,
)
from litagent.benchmark.survey_models import (
    SurveyBenchmarkCase,
    SurveyBenchmarkDataset,
    SurveyBenchmarkProfile,
    SurveyBenchmarkResult,
    SurveyCaseObservation,
    SurveyJudgeConfig,
    SurveyProfileComparison,
    SurveyProfileSummary,
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
from litagent.runner import LitAgent, RunPolicy

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


def runtime_prompt_source_fingerprint() -> str:
    """Hash the production files that define planning and report prompts."""
    root = Path(__file__).resolve().parents[1]
    return _hash_files(
        [
            root / "agents" / "planner.py",
            root / "agents" / "synthesis.py",
            root / "agents" / "reviewer.py",
            root / "agents" / "adversarial.py",
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
        by_cohort[(observation.case_id, observation.repetition)][
            observation.profile_id
        ] = observation
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
        ]
        comparable = bool(shared) and not mismatches
        left = by_profile[left_id]
        right = by_profile[right_id]
        deltas: dict[str, float | None] = {}
        if comparable:
            for field in fields:
                left_value = getattr(left, field)
                right_value = getattr(right, field)
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
    ) -> None:
        """Store configuration and optional offline execution boundary."""
        self._base_config = base_config
        self._judge_config = judge_config
        self._artifacts = artifacts
        self._run_archive = run_archive
        self._case_executor = case_executor or self._run_live_case
        self._uses_live_executor = case_executor is None
        self._judged_paper_ids: set[str] = set()

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
        dataset_fingerprint = dataset.fingerprint
        manifest_fingerprint = _hash_files(
            [Path(path) for path in dataset.manifest_paths]
        )
        for case in cases:
            for profile in profiles:
                config = self._config_for(profile, dataset.dataset_version)
                try:
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
                        "git_sha": git_sha,
                        "git_dirty": git_dirty,
                        "config_fingerprint": build_config_summary(config)[
                            "fingerprint"
                        ],
                        "profile_fingerprint": profile.fingerprint,
                        "prompt_source_fingerprint": prompt_fingerprint,
                        "dataset_fingerprint": dataset_fingerprint,
                        "manifest_fingerprint": manifest_fingerprint,
                    }
                )
                observations.append(observation)
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
        if comparable_first:
            _, first_summaries = aggregate_profiles(comparable_first)
        else:
            _, first_summaries = aggregate_profiles(first_pass)
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
        case_aggregates, summaries = aggregate_profiles(observations)
        eligible_summaries = [
            summary for summary in summaries if summary.profile_id in set(top_ids)
        ]
        recommended = (
            max(eligible_summaries, key=_rank_key).profile_id
            if eligible_summaries
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
            recommended_profile_id=profile.profile_id,
            git_sha=git_sha,
            git_dirty=git_dirty,
            reason_codes=[],
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
        incomplete = any(
            item.status != "completed"
            or item.metrics is None
            or item.judge is None
            or item.judge.status != "completed"
            for item in observations
        )
        status = "partial" if incomplete or reason_codes else "succeeded"
        formal_eligible = (
            status == "succeeded"
            and dataset.judgment_status in BASELINE_ELIGIBLE_JUDGMENT_STATUSES
            and not self_judged
        )
        final_reasons = list(reason_codes)
        if self_judged:
            final_reasons.append("self_judged")
        if dataset.judgment_status not in BASELINE_ELIGIBLE_JUDGMENT_STATUSES:
            final_reasons.append("dataset_judgment_not_approved")
        elif dataset.judgment_status == "owner_approved_ai_assisted":
            final_reasons.append("owner_approved_ai_assisted_ground_truth")
        if incomplete:
            final_reasons.append("incomplete_observations")
        manifest_paths = [Path(path) for path in dataset.manifest_paths]
        config_summary = build_config_summary(self._base_config)
        profile_comparisons = _profile_comparisons(observations, summaries)
        return SurveyBenchmarkResult(
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
                for comparison in profile_comparisons
                if comparison.comparable
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
    ) -> SurveyCaseObservation:
        """Run one real Survey, persist it, then invoke the independent Judge."""
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
            ) as agent:
                report = await agent.run(case.query)
            artifact = recorder.finalize(report, terminal_status="completed")
        except Exception as exc:
            error = f"survey_run_failed:{type(exc).__name__}"
            recorder.finalize(error=error, terminal_status="failed")
            artifact_path = self._run_archive.path_for(run_id)
            return SurveyCaseObservation(
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

        artifact_path = self._run_archive.path_for(run_id)
        metrics = evaluate_survey_artifact(
            case=case,
            artifact=artifact,
            judged_paper_ids=self._judged_paper_ids,
        )
        judge_trace = LangFuseTracer(
            host=config.observability.langfuse_host,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            trace_seed=f"{run_id}:judge",
        )
        judge_hook = RedactingTraceHook(
            judge_trace,
            payload_mode=config.observability.payload_mode,
        )
        judge_hook("survey.start", {"query": case.query, "session_id": run_id})
        judge_client = OpenAICompatibleClient(
            base_url=self._judge_config.base_url,
            model=self._judge_config.model,
            max_tokens=self._judge_config.max_tokens,
            temperature=0,
            trace_hook=judge_hook,
            api_key=os.getenv(self._judge_config.api_key_env, ""),
        )
        judge, judge_usage = await SurveyBenchmarkJudge(
            judge_client, self._judge_config
        ).evaluate(
            case=case,
            survey=str((artifact.get("report") or {}).get("survey") or ""),
            ledger=artifact_evidence_ledger(artifact),
        )
        judge_hook(
            "survey.complete",
            {
                "quality_status": (
                    "passed" if judge.status == "completed" else "unverified"
                ),
                "total_tokens": judge_usage.total_tokens,
                "delivery_status": metrics.delivery_status,
            },
        )
        judge_hook.flush()
        return SurveyCaseObservation(
            run_id=run_id,
            case_id=case.case_id,
            domain=case.domain,
            profile_id=profile.profile_id,
            repetition=repetition,
            status="completed",
            metrics=metrics,
            judge=judge,
            judge_usage=judge_usage,
            artifact_id=run_id,
            artifact_path=str(artifact_path),
            artifact_sha256=(
                f"sha256:{hashlib.sha256(artifact_path.read_bytes()).hexdigest()}"
            ),
            survey_trace_id=survey_tracer.trace_id,
            survey_trace_url=survey_tracer.trace_url,
            judge_trace_id=judge_trace.trace_id,
            judge_trace_url=judge_trace.trace_url,
            external_input_fingerprint=external_input_fingerprint(artifact),
            reason_codes=list(judge.reason_codes),
        )
