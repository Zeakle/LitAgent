"""Rebuild a failed public preparation locally without repeating paid extraction."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from litagent.agents.extraction_strategy import LLMStrategy, build_extraction_context
from litagent.agents.extractor import ExtractorWorker
from litagent.config import load_config
from litagent.context.templates import wrap_xml
from litagent.evidence import collect_ledger
from litagent.llm.client import LLMResponse
from litagent.observability.context import reset_task_id, set_task_id
from litagent.orchestrator.task_graph import SubTask, TaskGraph, TaskStatus
from litagent.quality import build_fact_cards
from litagent.rag.models import RawPaperAsset, SourceKind, SourceRef
from litagent.rag.pdf_parser import PyMuPDFParser
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget

VERSION = "received-preparation-rebuild-v1"


def file_sha(path):
    """Verify exact local bytes before recovering source positions."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_cached_paper(paper, expected_message, config):
    """Recover cached PDF metadata only when it reproduces the original model input."""
    candidate = copy.deepcopy(paper)
    candidate["quality_policy"] = "strict_auto_v2"
    context, _ = build_extraction_context(candidate)
    if wrap_xml("paper", context) == expected_message:
        return candidate, {}
    pid = candidate["paper_id"]
    if re.fullmatch(r"arxiv:\d{4}\.\d{4,5}(?:v\d+)?", pid) is None:
        raise ValueError("preparation_original_source_unrecoverable")
    name = pid.replace(":", "_")
    pdf = Path(config.rag.raw_root) / (name + ".pdf")
    audit = Path(config.rag.raw_root).parent / "parsed" / (name + ".json")
    record = json.loads(audit.read_text())
    asset_hash = file_sha(pdf)
    if record["paper_id"] != pid or record["asset_hash"] != asset_hash:
        raise ValueError("preparation_cached_asset_changed")
    asset = RawPaperAsset(
        paper_id=pid,
        title=candidate["title"],
        abstract=candidate.get("abstract") or "",
        asset_hash=asset_hash,
        pdf_path=pdf,
        sources=[
            SourceRef(
                kind=SourceKind.ARXIV_PDF,
                source_id=pid.removeprefix("arxiv:"),
                uri="https://arxiv.org/pdf/" + pid.removeprefix("arxiv:"),
                sha256=asset_hash,
            )
        ],
    )
    parsed = PyMuPDFParser().parse_with_quality(asset)
    if parsed.record is None:
        raise ValueError("preparation_cached_asset_quarantined")
    fulltext = [
        c.model_dump(mode="json")
        for c in parsed.record.chunks
        if c.content_scope.value == "selected_fulltext"
    ]
    candidate["chunks"] = [*candidate.get("chunks", []), *fulltext]
    candidate["content_scope"] = "selected_fulltext"
    candidate["provenance"] = [
        *candidate.get("provenance", []),
        *[s.model_dump(mode="json") for s in parsed.record.sources],
    ]
    context, _ = build_extraction_context(candidate)
    if wrap_xml("paper", context) != expected_message:
        raise ValueError("preparation_original_context_changed")
    candidate["fulltext_acquisition"] = {
        "status": "recovered_exact_cached_input",
        "asset_sha256": asset_hash,
        "scope": "selected_fulltext",
        "full_document_reviewed": False,
    }
    return candidate, {
        str(pdf.resolve()): asset_hash,
        str(audit.resolve()): file_sha(audit),
    }


def known_preparation_usage(source):
    """Refuse unknown, duplicated, or non-preparation operations before reuse."""
    opened, completed = {}, {}
    for event in source["events"]:
        kind, d = event["event"], event["data"]
        if kind == "llm.failed":
            raise ValueError("preparation_usage_unknown")
        if kind == "llm.start":
            op = d.get("operation_id")
            if (
                not op
                or op in opened
                or d.get("task_id") not in {"extract", "query_decomposition"}
            ):
                raise ValueError("preparation_operation_invalid")
            opened[op] = d
        elif kind == "llm.complete":
            op = d.get("operation_id")
            if (
                op not in opened
                or op in completed
                or d.get("usage_available") is not True
            ):
                raise ValueError("preparation_usage_invalid")
            if (
                d.get("actual_model") != "doubao-seed-2-1-lite-260915"
                or opened[op].get("model") != d.get("actual_model")
                or opened[op].get("task_id") != d.get("task_id")
                or d.get("finish_reason") != "stop"
            ):
                raise ValueError("preparation_model_or_finish_invalid")
            if (
                any(
                    type(d.get(k)) is not int or d[k] < 0
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                or d["total_tokens"] != d["prompt_tokens"] + d["completion_tokens"]
            ):
                raise ValueError("preparation_usage_invalid")
            completed[op] = d
    if not opened or set(opened) != set(completed) or len(opened) > 64:
        raise ValueError("preparation_incomplete_operations")
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    for d in completed.values():
        token = set_task_id(d["task_id"])
        try:
            budget.begin_request()
            budget.record(
                {
                    "prompt_tokens": d["prompt_tokens"],
                    "completion_tokens": d["completion_tokens"],
                }
            )
        finally:
            reset_task_id(token)
    if (
        source.get("usage", {}).get("total_tokens") != budget.used
        or source.get("usage", {}).get("prompt_tokens")
        != sum(d["prompt_tokens"] for d in completed.values())
        or source.get("usage", {}).get("completion_tokens")
        != sum(d["completion_tokens"] for d in completed.values())
    ):
        raise ValueError("preparation_accounting_does_not_reconcile")
    return opened, completed, budget.snapshot()


async def rebuild(source_path: Path, config):
    """Validate every received output and create an explicitly partial frozen graph."""
    if config.extractor.record_identity != "source_fields_v1":
        raise ValueError("preparation_rebuild_contract_not_enabled")
    source = json.loads(source_path.read_text())
    failure = source["graph"]["tasks"].get("extract", {})
    if (
        failure.get("status") != "failed"
        or failure.get("error") != "extraction_contract_invalid:duplicate_record"
    ):
        raise ValueError("preparation_parent_failure_invalid")
    opened, completed, budget = known_preparation_usage(source)
    papers = source["nodes"]["worker:relevance_gate"]["output"]
    extractions, seen_papers, assets = [], set(), {}
    for op, response in completed.items():
        if response["task_id"] != "extract":
            continue
        message = opened[op]["messages"][1]["content"]
        title = message.split("Title: ", 1)[1].split("\n\n[CHUNK:", 1)[0]
        matches = [p for p in papers if str(p.get("title", ""))[:500] == title]
        if len(matches) != 1 or matches[0]["paper_id"] in seen_papers:
            raise ValueError("preparation_paper_identity_ambiguous")
        paper, hashes = await asyncio.to_thread(
            restore_cached_paper, matches[0], message, config
        )
        assets.update(hashes)
        seen_papers.add(paper["paper_id"])
        # Replay only the SDK result; validation, worker projection, and IDs are real.
        llm, skills, loader = AsyncMock(), Mock(), Mock()
        skills.to_metadata_text_for.return_value = ""
        llm.chat.return_value = LLMResponse(
            content=response["content"],
            model=response["actual_model"],
            usage={
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "usage_available": True,
            },
            finish_reason=response["finish_reason"],
        )
        loader.enrich = AsyncMock(return_value=[paper])
        worker = ExtractorWorker(
            LLMStrategy(llm, skills, record_identity="source_fields_v1"),
            fulltext_loader=loader,
        )
        task = SubTask(
            "extract",
            "Replay received public extraction",
            "extractor",
            input_data={
                "quality_policy": "strict_auto_v2",
                "upstream_results": {"relevance_gate": [paper]},
            },
        )
        extractions.extend(await worker.execute(task))
    by_paper = {item["paper_id"]: item for item in extractions}
    # Model responses finish out of order; retain the original candidate ranking.
    extractions = [by_paper[p["paper_id"]] for p in papers if p["paper_id"] in by_paper]
    graph = TaskGraph()
    for tid, raw in source["graph"]["tasks"].items():
        if tid == "extract":
            result, status = extractions, TaskStatus.DONE
        elif raw["status"] == "done":
            result, status = source["nodes"]["worker:" + tid]["output"], TaskStatus.DONE
        elif tid == "adversarial_review" and raw["status"] == "skipped":
            result, status = None, TaskStatus.PENDING
        else:
            raise ValueError("preparation_unexpected_task_state")
        task = SubTask(
            tid,
            raw["description"],
            raw["agent_type"],
            input_data=copy.deepcopy(raw.get("input") or {}),
            status=status,
            result=result,
            timeout_ms=raw["timeout_ms"],
            max_retries=0,
        )
        graph.add_task(task, source["graph"]["dependencies"].get(tid, []))
    ledger = collect_ledger(extractions)
    output = {
        "version": VERSION,
        "source_run_id": source["run_id"],
        "source_artifact": str(source_path.resolve()),
        "source_artifact_sha256": file_sha(source_path),
        "query": source["query"],
        "graph": graph.to_snapshot(),
        "extractions": extractions,
        "evidence_sha256": digest(ledger),
        "fact_cards": build_fact_cards(ledger),
        "inherited_budget": budget,
        "asset_hashes": assets,
        "scope": {
            "relevant_papers": len(papers),
            "completed_papers": len(seen_papers),
            "not_extracted_paper_ids": [
                p["paper_id"] for p in papers if p["paper_id"] not in seen_papers
            ],
            "scientific_verdict": "not_verified",
            "source_failure_retained": True,
        },
        "search_source_outcomes": source["report"]["metadata"][
            "search_source_outcomes"
        ],
    }
    output["binding"] = digest(output)
    return output


async def main(args):
    """Write a new local reconstruction; never overwrite a paid parent."""
    if args.output.exists():
        raise ValueError("preparation_rebuild_output_exists")
    result = await rebuild(args.source, load_config())
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "binding": result["binding"],
                "scope": result["scope"],
                "cards": len(result["fact_cards"]),
                "inherited_budget": result["inherited_budget"],
                "paid_requests": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
