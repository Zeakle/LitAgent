"""Replay historical relevance candidates against verified benchmark abstracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from langchain_core.documents import Document

from litagent.agents.relevance_gate import build_relevance_text
from litagent.benchmark.artifacts import _atomic_write
from litagent.benchmark.datasets import load_survey_dataset
from litagent.benchmark.survey_models import SurveyBenchmarkResult
from litagent.rag.interfaces import ScoredDoc
from litagent.rag.reranker import CrossEncoderReranker


def _rank(reranker, query: str, papers: list[dict]) -> dict[str, int]:
    docs = [
        ScoredDoc(
            doc=Document(
                page_content=build_relevance_text(paper),
                metadata={"paper_id": paper["paper_id"]},
            ),
            score=0.0,
        )
        for paper in papers
    ]
    result = reranker.rerank(query, docs)
    return {
        item.doc.metadata["paper_id"]: index for index, item in enumerate(result, 1)
    }


def main() -> None:
    """Write a standalone ranking diagnostic without changing product artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("benchmarks/survey/dataset.yaml")
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = SurveyBenchmarkResult.model_validate_json(args.result.read_bytes())
    dataset = load_survey_dataset(args.dataset)
    abstracts = {}
    for manifest_path in args.manifest:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        for paper in manifest["papers"]:
            if paper.get("abstract_source") == "arxiv_api":
                digest = hashlib.sha256(
                    paper["abstract"].strip().encode("utf-8")
                ).hexdigest()
                if digest != paper.get("abstract_sha256"):
                    raise ValueError(f"unverified abstract: {paper['paper_id']}")
                abstracts[paper["paper_id"]] = paper
    reranker = CrossEncoderReranker()
    observations = []
    for item in result.observations:
        raw_path = Path(item.artifact_path.replace("\\", "/"))
        path = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
        artifact = json.loads(path.read_text(encoding="utf-8"))
        old = artifact.get("nodes", {}).get("worker:dedup", {}).get("output") or []
        selected = (
            artifact.get("nodes", {}).get("worker:relevance_gate", {}).get("output")
            or []
        )
        if not old:
            continue
        updated = [
            (
                {**paper, "abstract": abstracts[paper["paper_id"]]["abstract"]}
                if paper.get("paper_id") in abstracts
                else paper
            )
            for paper in old
        ]
        case = dataset.case_by_id(item.case_id)
        old_ranks = _rank(reranker, case.query, old)
        new_ranks = _rank(reranker, case.query, updated)
        selected_ids = {paper.get("paper_id") for paper in selected}
        original_ranks = {
            paper.get("paper_id"): paper.get("relevance_rank", index)
            for index, paper in enumerate(selected, 1)
        }
        cap = int(artifact["config"]["effective"]["relevance"]["max_papers"])
        observations.append(
            {
                "run_id": item.run_id,
                "case_id": item.case_id,
                "profile_id": item.profile_id,
                "method": reranker.model_name,
                "max_papers": cap,
                "candidates": [
                    {
                        "paper_id": paper["paper_id"],
                        "required": paper["paper_id"] in case.required_paper_ids,
                        "abstract_sha256": abstracts.get(paper["paper_id"], {}).get(
                            "abstract_sha256"
                        ),
                        "original_rank": original_ranks.get(paper["paper_id"]),
                        "old_replay_rank": old_ranks.get(paper["paper_id"]),
                        "verified_abstract_rank": new_ranks.get(paper["paper_id"]),
                        "historically_selected": paper["paper_id"] in selected_ids,
                        "new_rank_within_cap": new_ranks.get(paper["paper_id"], cap + 1)
                        <= cap,
                        "exclusion_reason": (
                            "rank_cap"
                            if new_ranks.get(paper["paper_id"], cap + 1) > cap
                            else None
                        ),
                    }
                    for paper in old
                ],
            }
        )
    payload = {
        "type": "offline_relevance_diagnostic",
        "source_run_id": result.run_id,
        "source_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "model": reranker.model_name,
        "interpretation": "Candidate reranking only; no new Survey or quality result",
        "observations": observations,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    _atomic_write(args.output, json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {len(observations)} offline observations to {args.output}")


if __name__ == "__main__":
    main()
