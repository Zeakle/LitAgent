"""Persist independent product and Judge stages without regenerating paid work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.survey_models import SurveyCaseObservation


def recovery_key(payload: Any) -> str:
    """Bind a checkpoint to canonical, non-secret content."""
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def verified_product(observation: SurveyCaseObservation) -> dict[str, Any]:
    """Read the exact recorded product bytes or fail without generating a replacement."""
    if not observation.artifact_path or not observation.artifact_sha256:
        raise ValueError("product_artifact_identity_missing")
    raw = Path(observation.artifact_path).read_bytes()
    if f"sha256:{hashlib.sha256(raw).hexdigest()}" != observation.artifact_sha256:
        raise ValueError("product_artifact_sha_mismatch")
    artifact = json.loads(raw)
    if isinstance(artifact, dict) and artifact.get("run_id") != (
        observation.artifact_id or observation.run_id
    ):
        raise ValueError("product_artifact_run_id_mismatch")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("report"), dict):
        raise ValueError("product_artifact_report_missing")
    if not str(artifact["report"].get("survey") or "").strip():
        raise ValueError("product_artifact_report_empty")
    return artifact


def validated_judge_calls(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject corrupt accounting fields instead of treating them as free attempts."""
    calls = state.get("calls")
    if not isinstance(calls, list):
        raise ValueError("judge_checkpoint_calls_invalid")
    for call in calls:
        if not isinstance(call, dict) or not isinstance(
            call.get("usage_complete"), bool
        ):
            raise ValueError("judge_checkpoint_usage_invalid")
        for key in ("prompt_tokens", "completion_tokens"):
            value = call.get(key)
            if value is None and not call["usage_complete"]:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("judge_checkpoint_tokens_invalid")
    return calls


class SurveyRecoveryStore:
    """Keep stage identities and cumulative accounting in one output root."""

    def __init__(self, repository: BenchmarkArtifactRepository) -> None:
        """Borrow the repository without allocating network or model resources."""
        self.repository = repository

    def product(self, slot: str, identity: str) -> SurveyCaseObservation | None:
        """Refuse incompatible or interrupted work instead of treating it as a miss."""
        saved = self.repository.read_checkpoint("survey-products", slot, strict=True)
        if saved is None:
            for legacy in self.repository.list_checkpoints("survey-observations"):
                legacy_slot = recovery_key(
                    {
                        key: legacy.get(key)
                        for key in ("case_id", "profile_id", "repetition")
                    }
                )
                if legacy_slot == slot:
                    raise ValueError(
                        "legacy_observation_requires_explicit_judge_only_import"
                    )
            return None
        if saved.get("version") != 1 or not isinstance(saved.get("identity"), str):
            raise ValueError("product_checkpoint_schema_invalid")
        if saved["identity"] != identity:
            raise ValueError("product_checkpoint_incompatible")
        if saved.get("observation") is None:
            raise ValueError("product_checkpoint_interrupted")
        observation = SurveyCaseObservation.model_validate(saved["observation"])
        verified_product(observation)
        return observation

    def save_product(
        self, slot: str, identity: str, observation: SurveyCaseObservation | None
    ) -> None:
        """Persist an in-flight reservation or the completed product before judging."""
        self.repository.write_checkpoint(
            "survey-products",
            slot,
            {
                "version": 1,
                "identity": identity,
                "observation": (
                    observation.model_dump(mode="json") if observation else None
                ),
            },
        )

    def judge(self, key: str) -> dict[str, Any]:
        """Return durable inventory, attempts, and optional terminal metrics."""
        saved = self.repository.read_checkpoint("survey-judges", key, strict=True)
        if saved is None:
            return {}
        validated_judge_calls(saved)
        if not isinstance(saved.get("binding"), str) or not isinstance(
            saved.get("aliases"), dict
        ):
            raise ValueError("judge_checkpoint_schema_invalid")
        return saved

    def save_judge(self, key: str, state: dict[str, Any]) -> None:
        """Commit stage progress immediately after each provider response."""
        self.repository.write_checkpoint("survey-judges", key, state)

    def accounting(self) -> dict[str, Any]:
        """Sum unique product work and every Judge attempt, including older sessions."""
        products = self.repository.list_checkpoints("survey-products")
        judges = self.repository.list_checkpoints("survey-judges")
        seen = set()
        product_tokens = 0
        incomplete = False
        for saved in products:
            raw = saved.get("observation")
            if raw is None:
                incomplete = True
                continue
            item = SurveyCaseObservation.model_validate(raw)
            if item.run_id in seen:
                continue
            seen.add(item.run_id)
            if item.metrics is None:
                incomplete = True
            else:
                product_tokens += item.metrics.cost.total_tokens
                incomplete |= not item.metrics.cost.usage_complete
        calls = [call for saved in judges for call in validated_judge_calls(saved)]
        accounted_traces = {
            trace for saved in judges for trace in saved.get("trace_history", [])
        }
        legacy_judge_tokens = 0
        for raw in self.repository.list_checkpoints("survey-observations"):
            item = SurveyCaseObservation.model_validate(raw)
            if item.run_id not in seen:
                seen.add(item.run_id)
                if item.metrics is None:
                    incomplete = True
                else:
                    product_tokens += item.metrics.cost.total_tokens
                    incomplete |= not item.metrics.cost.usage_complete
            # Copies of an observation must not multiply the same historical bill.
            trace = item.judge_trace_id
            if trace and trace not in accounted_traces:
                accounted_traces.add(trace)
                legacy_judge_tokens += item.judge_usage.total_tokens
                incomplete |= item.judge_usage.usage_complete is not True
        return {
            "products": len(seen)
            + sum(item.get("observation") is None for item in products),
            "product_tokens": product_tokens,
            "judge_tokens": sum(
                call.get("prompt_tokens", 0) + call.get("completion_tokens", 0)
                for call in calls
            )
            + legacy_judge_tokens,
            "judge_calls": len(calls),
            "usage_complete": not incomplete
            and all(call.get("usage_complete", False) for call in calls),
        }
