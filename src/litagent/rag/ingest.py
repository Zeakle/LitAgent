"""Orchestrate manifest ingestion outside the Survey task graph."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litagent.exceptions import ConfigError
from litagent.rag.corpus import PaperSyncResult
from litagent.rag.manifest import (
    ManifestValidationError,
    load_manifest,
    materialize_manifest_assets,
)
from litagent.rag.models import PaperRecord, merge_paper_records
from litagent.rag.pdf_parser import CorpusParseError, metadata_completeness
from litagent.rag.quality import (
    DocumentQualityMetrics,
    DocumentQualityReport,
    IngestionOutcome,
    PaperIngestionReport,
    QualityDecision,
)
from litagent.rag.state import IngestionStatus


@dataclass(frozen=True)
class IngestionSummary:
    """Summarize one resumable manifest ingestion batch."""

    batch_id: str
    status: str
    paper_count: int
    succeeded_count: int
    failed_count: int
    embedded_count: int
    payload_updated_count: int
    deleted_count: int
    unchanged_count: int
    reason_codes: list[str]
    reports: list[PaperIngestionReport]

    @classmethod
    def from_results(
        cls,
        batch_id: str,
        results,
        reports: list[PaperIngestionReport] | None = None,
    ) -> "IngestionSummary":
        """Build an ingestion summary from paper results and reports."""
        results = list(results)
        paper_ids = {result.paper_id for result in results}
        failed = [
            result for result in results if result.status is IngestionStatus.FAILED
        ]
        failed_paper_ids = {result.paper_id for result in failed}
        return cls(
            batch_id=batch_id,
            status="failed" if failed else "succeeded",
            # One paper can produce both a quarantine decision and a prune
            # result. Count unique papers while retaining all operation totals.
            paper_count=len(paper_ids),
            succeeded_count=len(paper_ids - failed_paper_ids),
            failed_count=len(failed_paper_ids),
            embedded_count=sum(item.embedded_count for item in results),
            payload_updated_count=sum(item.payload_updated_count for item in results),
            deleted_count=sum(item.deleted_count for item in results),
            unchanged_count=sum(item.unchanged_count for item in results),
            reason_codes=list(
                dict.fromkeys(item.reason_code for item in failed if item.reason_code)
            ),
            reports=reports or [],
        )


class QuarantineRepository:
    """Persist only safe failure metadata beside ignored corpus assets."""

    def __init__(self, root: Path) -> None:
        """Initialize the quarantine repository."""
        self._root = root

    async def record(self, asset, *, reason_code: str) -> None:
        """Atomically save a local diagnostic without PDF bytes or prompts."""
        self._root.mkdir(parents=True, exist_ok=True)
        safe_id = asset.paper_id.replace(":", "_").replace("/", "_")
        target = self._root / f"{safe_id}.json"
        temporary = target.with_suffix(".json.tmp")
        payload = {
            "paper_id": asset.paper_id,
            "title": asset.title,
            "asset_hash": asset.asset_hash,
            "source_kinds": [source.kind.value for source in asset.sources],
            "reason_code": reason_code,
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def list_entries(self) -> list[dict]:
        """Return valid quarantine summaries without failing on corrupt files."""
        entries = []
        if not self._root.exists():
            return entries
        for path in sorted(self._root.glob("*.json")):
            try:
                entries.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return entries


class ParsedAuditRepository:
    """Persist local-only quality details excluded from external telemetry."""

    def __init__(self, root: Path) -> None:
        """Initialize the parsed audit repository."""
        self._root = root.resolve()

    async def record(self, asset, parsed) -> dict[str, Any]:
        """Persist one parsed-document audit record."""
        self._root.mkdir(parents=True, exist_ok=True)
        safe_id = asset.paper_id.replace(":", "_").replace("/", "_")
        target = self._root / f"{safe_id}.json"
        temporary = target.with_suffix(".json.tmp")
        payload = {
            "paper_id": asset.paper_id,
            "asset_hash": asset.asset_hash,
            "quality": parsed.quality.model_dump(mode="json"),
            "excluded_blocks": [
                block.model_dump(mode="json") for block in parsed.audit.excluded_blocks
            ],
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, target)
        return {
            "paper_id": asset.paper_id,
            "excluded_count": len(parsed.audit.excluded_blocks),
            "reason_codes": parsed.quality.reason_codes,
        }


def build_ingestion_report(
    *,
    paper_id: str,
    quality: DocumentQualityReport,
    sync_result: PaperSyncResult | None,
    has_selected_fulltext: bool,
    elapsed_ms: int,
) -> PaperIngestionReport:
    """Map quality and storage results into one explainable paper outcome."""
    if quality.decision is QualityDecision.QUARANTINED:
        outcome = IngestionOutcome.QUARANTINED
        storage_status = None
    elif sync_result is None or sync_result.status is IngestionStatus.FAILED:
        outcome = IngestionOutcome.FAILED
        storage_status = "failed"
    elif has_selected_fulltext:
        outcome = IngestionOutcome.INDEXED
        storage_status = "succeeded"
    else:
        outcome = IngestionOutcome.METADATA_ONLY
        storage_status = "succeeded"
    reasons = [*quality.reason_codes]
    if sync_result and sync_result.reason_code:
        reasons.append(sync_result.reason_code)
    return PaperIngestionReport(
        paper_id=paper_id,
        outcome=outcome,
        storage_status=storage_status,
        reason_codes=list(dict.fromkeys(reasons)),
        warnings=quality.warnings,
        metrics=quality.metrics,
        elapsed_ms=elapsed_ms,
    )


class CorpusIngestor:
    """Load one manifest and synchronize normalized records outside the DAG."""

    def __init__(
        self,
        *,
        config,
        service,
        parser,
        local_pdf_adapter,
        pdf_adapter,
        quarantine,
        audit_repository=None,
    ) -> None:
        """Initialize the corpus ingestor."""
        self._config = config
        self._service = service
        self._parser = parser
        self._local_pdf_adapter = local_pdf_adapter
        self._pdf_adapter = pdf_adapter
        self._quarantine = quarantine
        self._audit = audit_repository or ParsedAuditRepository(
            Path(getattr(config, "parsed_root", "artifacts/corpus/parsed"))
        )

    async def ingest_manifest(
        self,
        manifest_path: Path,
        *,
        resume: bool = False,
    ) -> IngestionSummary:
        """Parse and sync a manifest; resume safely replays idempotent work."""
        manifest = load_manifest(
            manifest_path,
            raw_root=Path(self._config.raw_root),
        )
        if manifest.corpus_version != self._config.corpus_version:
            raise ConfigError(
                "manifest corpus_version does not match rag.corpus_version"
            )
        batch_id = uuid.uuid4().hex
        results: list[PaperSyncResult] = []
        reports: list[PaperIngestionReport] = []
        assets = materialize_manifest_assets(manifest)
        active_paper_ids = {asset.paper_id for asset in assets}
        for asset in assets:
            started = time.monotonic()
            quality = DocumentQualityReport(
                decision=QualityDecision.ACCEPTED,
                metrics=DocumentQualityMetrics(
                    metadata_completeness=metadata_completeness(asset)
                ),
            )
            try:
                abstract_record = PaperRecord.from_abstract_asset(asset)
                if self._config.content_mode == "abstract_and_selected_fulltext":
                    if asset.pdf_url and asset.pdf_path is None:
                        asset = await self._pdf_adapter.materialize(asset)
                    elif asset.pdf_path is not None:
                        asset = await self._local_pdf_adapter.materialize(asset)
                    if asset.pdf_path is not None:
                        parsed = self._parser.parse_with_quality(asset)
                        # The full local audit is written before any cleaned
                        # content crosses the Qdrant storage boundary.
                        try:
                            await self._audit.record(asset, parsed)
                        except OSError as exc:
                            raise CorpusParseError(
                                "audit_write_failed",
                                "local parsed audit could not be persisted",
                            ) from exc
                        quality = parsed.quality
                        if parsed.record is None:
                            # Quarantine is authoritative: the final prune must
                            # remove any previously indexed unsafe version.
                            active_paper_ids.discard(asset.paper_id)
                            reason_code = (
                                quality.reason_codes[0]
                                if quality.reason_codes
                                else "document_quarantined"
                            )
                            await self._quarantine.record(
                                asset,
                                reason_code=reason_code,
                            )
                            sync_result = PaperSyncResult(
                                paper_id=asset.paper_id,
                                status=IngestionStatus.SUCCEEDED,
                                reason_code=reason_code,
                            )
                            results.append(sync_result)
                            reports.append(
                                build_ingestion_report(
                                    paper_id=asset.paper_id,
                                    quality=quality,
                                    sync_result=sync_result,
                                    has_selected_fulltext=False,
                                    elapsed_ms=int((time.monotonic() - started) * 1000),
                                )
                            )
                            continue
                        record = merge_paper_records([abstract_record, parsed.record])[
                            0
                        ]
                    else:
                        record = abstract_record
                else:
                    # Abstract mode intentionally avoids all PDF I/O while still
                    # producing a quality report for the unified CLI contract.
                    record = abstract_record

                sync_result = await self._service.sync_record(
                    record,
                    batch_id=batch_id,
                )
                results.append(sync_result)
                reports.append(
                    build_ingestion_report(
                        paper_id=asset.paper_id,
                        quality=quality,
                        sync_result=sync_result,
                        has_selected_fulltext=any(
                            chunk.content_scope.value == "selected_fulltext"
                            for chunk in record.chunks
                        ),
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                    )
                )
            except (CorpusParseError, ManifestValidationError) as exc:
                await self._quarantine.record(
                    asset,
                    reason_code=exc.code,
                )
                sync_result = PaperSyncResult(
                    paper_id=asset.paper_id,
                    status=IngestionStatus.FAILED,
                    reason_code=exc.code,
                )
                results.append(sync_result)
                quality = quality.model_copy(
                    update={
                        "decision": QualityDecision.DEGRADED,
                        "reason_codes": list(
                            dict.fromkeys([*quality.reason_codes, exc.code])
                        ),
                    }
                )
                reports.append(
                    build_ingestion_report(
                        paper_id=asset.paper_id,
                        quality=quality,
                        sync_result=sync_result,
                        has_selected_fulltext=False,
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                    )
                )
        results.extend(
            await self._service.prune_missing(
                active_paper_ids,
                batch_id=batch_id,
            )
        )
        return IngestionSummary.from_results(batch_id, results, reports)
