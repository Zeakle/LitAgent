"""Serve packaged, explicitly public records without opening private runtime state."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from litagent.benchmark.survey_archive import SurveyBenchmarkArchive
from litagent.config import AppConfig, LLMConfig

PUBLIC_ROOT = Path(__file__).resolve().parent / "static" / "presentation"


def presentation_enabled() -> bool:
    """Identify the explicit process-wide, read-only presentation mode."""
    return os.getenv("LITAGENT_DEMO_MODE", "").lower() == "presentation"


def presentation_config() -> AppConfig:
    """Use public model labels without requiring a checkout or live config lookup."""
    models = json.loads((PUBLIC_ROOT / "suite.json").read_text(encoding="utf-8"))[
        "execution_context"
    ]["models"]
    return AppConfig(
        agent={},
        logging={},
        llm=LLMConfig(
            model=models["extraction"],
            **{
                f"{stage}_model": models[stage]
                for stage in ("extraction", "generation", "evaluation")
            },
        ),
    )


class PresentationRepository:
    """Read only the fixed packaged publication, never a client-supplied path."""

    def list(self) -> list[dict]:
        """Return copies so rendering cannot modify shared public records."""
        path = PUBLIC_ROOT / "runs.json"
        return copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))

    def get(self, run_id: str) -> dict | None:
        """Reject unknown identities without falling back to local archives."""
        return next((r for r in self.list() if r["run_id"] == run_id), None)


class PresentationService:
    """Satisfy archive/export reads without creating SQLite or task executors."""

    async def list(self) -> list:
        """Keep the interactive list empty in presentation mode."""
        return []

    async def get(self, run_id: str):
        """Let the existing version resolver use packaged archive records."""
        raise KeyError(run_id)

    async def get_view(self, run_id: str, **kwargs):
        """Return the normal missing-interactive-run response without private lookup."""
        raise KeyError(run_id)

    async def get_version(self, run_id: str, revision: int, **kwargs):
        """Preserve the existing archive fallback and exact revision checks."""
        raise KeyError(run_id)

    async def get_audit_artifact(self, run_id: str):
        """Never load private audit state."""
        raise KeyError(run_id)

    async def versions(self, run_id: str):
        """Leave archive version projection to the existing client contract."""
        raise KeyError(run_id)

    async def export(self, run_id: str, revision: int, format: str, **kwargs):
        """Use the existing archive formatter, including source-SHA headers."""
        raise ValueError("report_revision_not_found")


class PresentationBenchmarkArchive(SurveyBenchmarkArchive):
    """Reuse benchmark validation and UI with only packaged public inputs."""

    def __init__(self):
        """Restrict every inherited scan to the publication directory."""
        super().__init__(PUBLIC_ROOT, PUBLIC_ROOT / "runs", PUBLIC_ROOT / "audits")

    def observation(self, benchmark_id: str, observation_id: str) -> dict:
        """Return a sanitized replay rather than dereferencing private file paths."""
        result = self.get(benchmark_id)
        observation = next(
            (o for o in result.observations if o.get("run_id") == observation_id), None
        )
        artifact = PresentationRepository().get(observation_id)
        if observation is None or artifact is None:
            raise KeyError(observation_id)
        return {"observation": observation, "artifact": artifact, "public_replay": True}
