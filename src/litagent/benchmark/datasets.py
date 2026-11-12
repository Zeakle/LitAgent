"""Load versioned YAML benchmark inputs through strict Pydantic models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from litagent.benchmark.models import (
    IngestionFixtureCase,
    RAGBenchmarkDataset,
    RAGBenchmarkProfile,
)
from litagent.benchmark.survey_models import (
    SurveyBenchmarkDataset,
    SurveyBenchmarkProfile,
    SurveyJudgeConfig,
)


class IngestionFixtureDataset(BaseModel):
    """Validate the ingestion robustness fixture list."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    cases: list[IngestionFixtureCase] = Field(min_length=1)


class RAGProfileSet(BaseModel):
    """Validate an explicit profile list without generating products."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profiles: list[RAGBenchmarkProfile] = Field(min_length=1)


class SurveyProfileSet(BaseModel):
    """Validate the three explicit product-benchmark profiles."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    profiles: list[SurveyBenchmarkProfile] = Field(min_length=1)


def _load_yaml(path: Path) -> dict:
    """Load and validate a YAML mapping from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"benchmark input not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark input must be a mapping: {path}")
    return payload


def load_ingestion_dataset(path: Path) -> IngestionFixtureDataset:
    """Load and validate an ingestion benchmark dataset."""
    return IngestionFixtureDataset.model_validate(_load_yaml(path))


def load_retrieval_dataset(path: Path) -> RAGBenchmarkDataset:
    """Load and validate a retrieval benchmark dataset."""
    return RAGBenchmarkDataset.model_validate(_load_yaml(path))


def load_profiles(path: Path) -> list[RAGBenchmarkProfile]:
    """Load and validate benchmark profiles."""
    profile_set = RAGProfileSet.model_validate(_load_yaml(path))
    ids = [profile.profile_id for profile in profile_set.profiles]
    fingerprints = [profile.fingerprint for profile in profile_set.profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark profile_id values must be unique")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("benchmark profiles contain duplicate behavior")
    return profile_set.profiles


def load_survey_dataset(path: Path) -> SurveyBenchmarkDataset:
    """Load and validate the formal Survey benchmark dataset."""
    return SurveyBenchmarkDataset.model_validate(_load_yaml(path))


def load_survey_profiles(path: Path) -> list[SurveyBenchmarkProfile]:
    """Load unique Survey benchmark profiles."""
    profile_set = SurveyProfileSet.model_validate(_load_yaml(path))
    ids = [profile.profile_id for profile in profile_set.profiles]
    fingerprints = [profile.fingerprint for profile in profile_set.profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("survey benchmark profile_id values must be unique")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("survey benchmark profiles contain duplicate behavior")
    return profile_set.profiles


def load_survey_judge_config(path: Path) -> SurveyJudgeConfig:
    """Load a non-secret Judge configuration from YAML."""
    return SurveyJudgeConfig.model_validate(_load_yaml(path))
