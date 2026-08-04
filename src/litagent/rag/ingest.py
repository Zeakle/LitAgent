"""Orchestrate manifest ingestion outside the Survey task graph."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from litagent.exceptions import ConfigError
from litagent.rag.corpus import PaperSyncResult
from litagent.rag.manifest import ManifestValidationError, load_manifest
from litagent.rag.models import PaperRecord, merge_paper_records
from litagent.rag.pdf_parser import CorpusParseError
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

    @classmethod
    def from_results(cls, batch_id: str, results) -> "IngestionSummary":
        failed = [
            result for result in results if result.status is IngestionStatus.FAILED
        ]
        return cls(
            batch_id=batch_id,
            status="failed" if failed else "succeeded",
            paper_count=len(results),
            succeeded_count=len(results) - len(failed),
            failed_count=len(failed),
            embedded_count=sum(item.embedded_count for item in results),
            payload_updated_count=sum(item.payload_updated_count for item in results),
            deleted_count=sum(item.deleted_count for item in results),
            unchanged_count=sum(item.unchanged_count for item in results),
            reason_codes=list(
                dict.fromkeys(item.reason_code for item in failed if item.reason_code)
            ),
        )


class QuarantineRepository:
    """Persist only safe failure metadata beside ignored corpus assets."""

    def __init__(self, root: Path) -> None:
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
    ) -> None:
        self._config = config
        self._service = service
        self._parser = parser
        self._local_pdf_adapter = local_pdf_adapter
        self._pdf_adapter = pdf_adapter
        self._quarantine = quarantine

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
        records = []
        failures = []
        assets = manifest.to_assets()
        for asset in assets:
            try:
                abstract_record = PaperRecord.from_abstract_asset(asset)
                if self._config.content_mode == "abstract_and_selected_fulltext":
                    if asset.pdf_url and asset.pdf_path is None:
                        asset = await self._pdf_adapter.materialize(asset)
                    elif asset.pdf_path is not None:
                        asset = await self._local_pdf_adapter.materialize(asset)
                    if asset.pdf_path is not None:
                        records.extend(
                            merge_paper_records(
                                [abstract_record, self._parser.parse(asset)]
                            )
                        )
                    else:
                        records.append(abstract_record)
                else:
                    # Abstract mode intentionally does not read, hash, download,
                    # or parse PDF content. All sources converge to one abstract
                    # chunk contract; only provenance differs.
                    records.append(abstract_record)
            except CorpusParseError as exc:
                await self._quarantine.record(
                    asset,
                    reason_code=exc.code,
                )
                failures.append(
                    PaperSyncResult(
                        paper_id=asset.paper_id,
                        status=IngestionStatus.FAILED,
                        reason_code=exc.code,
                    )
                )
            except ManifestValidationError:
                await self._quarantine.record(
                    asset,
                    reason_code="source_validation_failed",
                )
                failures.append(
                    PaperSyncResult(
                        paper_id=asset.paper_id,
                        status=IngestionStatus.FAILED,
                        reason_code="source_validation_failed",
                    )
                )
        # Replaying every manifest entry is deliberate: deterministic hashes make
        # succeeded papers cheap no-ops and also cover interruption before a state
        # row was created.
        results = failures + [
            await self._service.sync_record(record, batch_id=batch_id)
            for record in merge_paper_records(records)
        ]
        results.extend(
            await self._service.prune_missing(
                {asset.paper_id for asset in assets},
                batch_id=batch_id,
            )
        )
        return IngestionSummary.from_results(batch_id, results)
