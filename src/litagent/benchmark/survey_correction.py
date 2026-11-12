"""Derive a non-destructive deterministic diagnostic from historical Survey runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from litagent.benchmark.artifacts import (
    BenchmarkArtifactPaths,
    BenchmarkArtifactRepository,
)
from litagent.benchmark.datasets import load_survey_dataset
from litagent.benchmark.survey_metrics import (
    METRICS_VERSION,
    aggregate_profiles,
    evaluate_survey_artifact,
)
from litagent.benchmark.survey_models import SurveyBenchmarkResult
from litagent.benchmark.survey_runner import _profile_comparisons


def derive_historical_survey_diagnostic(
    source: Path,
    dataset_path: Path,
    output_root: Path,
    *,
    project_root: Path,
) -> BenchmarkArtifactPaths:
    """Recompute only deterministic metrics; preserve original Judge decisions."""
    source_bytes = source.read_bytes()
    original = SurveyBenchmarkResult.model_validate_json(source_bytes)
    dataset = load_survey_dataset(dataset_path)
    if original.dataset_fingerprint != dataset.fingerprint:
        raise ValueError("historical_dataset_fingerprint_mismatch")
    runs_root = (project_root / "artifacts/runs").resolve()
    updated = []
    for observation in original.observations:
        if not observation.artifact_path or not observation.artifact_sha256:
            raise ValueError(f"missing_historical_artifact:{observation.run_id}")
        raw_path = Path(observation.artifact_path.replace("\\", "/"))
        path = (
            raw_path if raw_path.is_absolute() else project_root / raw_path
        ).resolve()
        if not path.is_relative_to(runs_root):
            raise ValueError(
                f"historical_artifact_outside_runs_root:{observation.run_id}"
            )
        content = path.read_bytes()
        expected = observation.artifact_sha256.removeprefix("sha256:")
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"historical_artifact_sha_mismatch:{observation.run_id}")
        artifact: dict[str, Any] = json.loads(content)
        if artifact.get("run_id") != observation.run_id:
            raise ValueError(f"historical_artifact_id_mismatch:{observation.run_id}")
        metrics = evaluate_survey_artifact(
            case=dataset.case_by_id(observation.case_id), artifact=artifact
        )
        # The old two-call RAGAS usage was not fully observable; do not infer
        # a cost advantage from product token totals that omit those calls.
        metrics.cost.usage_complete = False
        metrics.cost.missing_usage_calls = max(1, metrics.cost.missing_usage_calls)
        updated.append(observation.model_copy(update={"metrics": metrics}))
    _, summaries = aggregate_profiles(updated)
    comparisons = _profile_comparisons(updated, summaries)
    payload = {
        "benchmark_type": "survey_corrected",
        "run_id": f"{original.run_id}-corrected-v4",
        "source_run_id": original.run_id,
        "source_sha256": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
        "metrics_version": METRICS_VERSION,
        "dataset_fingerprint": original.dataset_fingerprint,
        "status": "diagnostic_only",
        "formal_eligible": False,
        "observations": [item.model_dump(mode="json") for item in updated],
        "profile_summaries": [item.model_dump(mode="json") for item in summaries],
        "profile_comparisons": [item.model_dump(mode="json") for item in comparisons],
        "limitations": [
            "old_external_inputs_ready_boundary_missing",
            "historical_ragas_usage_unrecoverable",
            "judge_metrics_not_recomputed",
            "not_formal_baseline",
        ],
    }
    destination = output_root / f"{payload['run_id']}.json"
    if destination.exists():
        raise FileExistsError("corrected diagnostic already exists")
    return BenchmarkArtifactRepository(output_root).write(payload)
