"""Run deterministic dirty-data cases without shared storage."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence

from litagent.benchmark.metrics import aggregate_ingestion_cases
from litagent.benchmark.models import (
    IngestionBenchmarkResult,
    IngestionCaseObservation,
    IngestionFixtureCase,
)

IngestionCaseExecutor = Callable[
    [IngestionFixtureCase],
    Awaitable[IngestionCaseObservation],
]


def _dataset_fingerprint(cases: Sequence[IngestionFixtureCase]) -> str:
    """Return a stable ingestion-dataset fingerprint."""
    payload = [case.model_dump(mode="json") for case in cases]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class IngestionRobustnessRunner:
    """Execute every case independently and preserve failed observations."""

    def __init__(self, executor: IngestionCaseExecutor) -> None:
        """Initialize the ingestion robustness runner."""
        self._executor = executor

    async def run(
        self,
        cases: Sequence[IngestionFixtureCase],
    ) -> IngestionBenchmarkResult:
        """Run the ingestion robustness runner workflow."""
        if not cases:
            raise ValueError("ingestion benchmark requires cases")
        ids = [case.case_id for case in cases]
        if len(set(ids)) != len(ids):
            raise ValueError("ingestion case_id values must be unique")

        observations: list[IngestionCaseObservation] = []
        for case in cases:
            started = time.perf_counter()
            try:
                observation = await self._executor(case)
                if observation.case_id != case.case_id:
                    raise ValueError("ingestion executor returned a different case_id")
            except Exception as exc:
                observation = IngestionCaseObservation(
                    case_id=case.case_id,
                    expected_outcome=case.expected_outcome,
                    actual_outcome="failed",
                    expected_reason_codes=case.expected_reason_codes,
                    actual_reason_codes=[
                        getattr(exc, "code", None) or type(exc).__name__
                    ],
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            observations.append(observation)

        return aggregate_ingestion_cases(
            observations,
            run_id=f"ingestion-{uuid.uuid4().hex}",
            dataset_fingerprint=_dataset_fingerprint(cases),
        )
