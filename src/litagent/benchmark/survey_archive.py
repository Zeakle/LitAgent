"""Read validated Survey benchmark archives without trusting client paths."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from litagent.benchmark.citation_audit import CitationAuditResult
from litagent.benchmark.survey_models import SurveyBenchmarkResult
from litagent.logging import get_logger

logger = get_logger("benchmark.survey_archive")


class SurveyBenchmarkArchive:
    """Index only complete result JSON and verify linked run artifacts."""

    def __init__(
        self,
        root: Path | str = "artifacts/benchmarks/survey",
        runs_root: Path | str = "artifacts/runs",
        citation_audit_root: Path | str = "artifacts/benchmarks/citation-audit",
    ) -> None:
        self.root = Path(root).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.citation_audit_root = Path(citation_audit_root).resolve()
        self.project_root = self.runs_root.parent.parent

    def _citation_audits(
        self,
        benchmark_id: str,
        observation_id: str,
        artifact_sha256: str,
    ) -> list[dict[str, Any]]:
        """Attach only sidecars that explicitly identify this recorded Survey run."""
        if not self.citation_audit_root.exists():
            return []
        linked: list[tuple[str, dict[str, Any]]] = []
        for path in self.citation_audit_root.rglob("*.json"):
            if "checkpoints" in path.relative_to(self.citation_audit_root).parts:
                continue
            if not path.resolve().is_relative_to(self.citation_audit_root):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, dict)
                    or payload.get("benchmark_type") != "citation_audit"
                ):
                    continue
                audit = CitationAuditResult.model_validate(payload)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid citation audit %s (%s)",
                    path,
                    type(exc).__name__,
                )
                continue
            if (
                audit.source_benchmark_id != benchmark_id
                or audit.source_run_id != observation_id
            ):
                continue
            item = audit.model_dump(mode="json")
            item["association_status"] = (
                "current"
                if audit.source_artifact_sha256 == artifact_sha256
                else "stale_diagnostic"
            )
            linked.append((audit.completed_at or audit.started_at, item))
        linked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in linked]

    def _index(self) -> tuple[dict[str, SurveyBenchmarkResult], set[str]]:
        results: dict[str, SurveyBenchmarkResult] = {}
        duplicates: set[str] = set()
        if not self.root.exists():
            return results, duplicates
        for path in self.root.rglob("*.json"):
            if set(path.relative_to(self.root).parts) & {"checkpoints", "corrected"}:
                continue
            if not path.resolve().is_relative_to(self.root):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, dict)
                    or payload.get("benchmark_type") != "survey"
                ):
                    continue
                result = SurveyBenchmarkResult.model_validate(payload)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid Survey benchmark %s (%s)",
                    path,
                    type(exc).__name__,
                )
                continue
            if result.run_id in results:
                duplicates.add(result.run_id)
            else:
                results[result.run_id] = result
        return results, duplicates

    def list(self) -> list[dict[str, Any]]:
        """Return summaries, excluding ambiguous archive identities."""
        results, duplicates = self._index()
        return [
            {
                "benchmark_id": result.run_id,
                "stage": result.stage,
                "status": result.status,
                "dataset_id": result.dataset_id,
                "dataset_version": result.dataset_version,
                "observation_count": len(result.observations),
                "profile_count": len(result.profile_summaries),
                "formal_eligible": result.formal_eligible,
                "self_judged": result.self_judged,
            }
            for result in results.values()
            if result.run_id not in duplicates
        ]

    def get(self, benchmark_id: str) -> SurveyBenchmarkResult:
        """Reject missing and duplicate IDs instead of choosing an arbitrary file."""
        results, duplicates = self._index()
        if benchmark_id in duplicates:
            raise ValueError("duplicate_benchmark_id")
        if benchmark_id not in results:
            raise KeyError(benchmark_id)
        return results[benchmark_id]

    def corrected_diagnostic(self, benchmark_id: str) -> dict[str, Any] | None:
        """Use a derived summary only when its source SHA matches the archive."""
        self.get(benchmark_id)
        if re.fullmatch(r"[A-Za-z0-9_-]+", benchmark_id) is None:
            return None
        source_paths = [
            path
            for path in self.root.rglob(f"{benchmark_id}.json")
            if "corrected" not in path.relative_to(self.root).parts
            and "checkpoints" not in path.relative_to(self.root).parts
            and path.resolve().is_relative_to(self.root)
        ]
        if len(source_paths) != 1:
            return None
        try:
            source_sha = (
                "sha256:" + hashlib.sha256(source_paths[0].read_bytes()).hexdigest()
            )
        except OSError:
            return None
        path = self.root / "corrected" / f"{benchmark_id}-corrected-v4.json"
        if not path.is_file():
            return None
        try:
            diagnostic = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(diagnostic, dict)
            or diagnostic.get("benchmark_type") != "survey_corrected"
            or diagnostic.get("source_run_id") != benchmark_id
            or diagnostic.get("source_sha256") != source_sha
        ):
            return None
        return diagnostic

    def observation(self, benchmark_id: str, observation_id: str) -> dict[str, Any]:
        """Open a recorded run only after root containment and SHA verification."""
        result = self.get(benchmark_id)
        matches = [
            item for item in result.observations if item.run_id == observation_id
        ]
        if not matches:
            raise KeyError(observation_id)
        if len(matches) != 1:
            raise ValueError("duplicate_observation_id")
        observation = matches[0]
        if not observation.artifact_path or not observation.artifact_sha256:
            raise ValueError("observation_artifact_unavailable")
        raw_path = Path(observation.artifact_path.replace("\\", "/"))
        path = (
            raw_path if raw_path.is_absolute() else self.project_root / raw_path
        ).resolve()
        if not path.is_relative_to(self.runs_root):
            raise ValueError("observation_artifact_outside_runs_root")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ValueError("observation_artifact_unavailable") from exc
        expected = observation.artifact_sha256.removeprefix("sha256:")
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("observation_artifact_sha_mismatch")
        try:
            artifact = json.loads(content)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("observation_artifact_invalid") from exc
        if (
            not isinstance(artifact, dict)
            or artifact.get("run_id") != observation.run_id
        ):
            raise ValueError("observation_artifact_identity_mismatch")
        return {
            "observation": observation.model_dump(mode="json"),
            "artifact": artifact,
            "citation_audits": self._citation_audits(
                benchmark_id,
                observation.run_id,
                observation.artifact_sha256,
            ),
        }
