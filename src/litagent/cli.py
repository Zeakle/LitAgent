"""Run survey, configuration, and tool-listing commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from litagent.config import load_config
from litagent.runner import LitAgent, derive_delivery


def _add_benchmark_commands(subparsers) -> None:
    """Register offline ingestion and explicit live retrieval benchmarks."""
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run reproducible ingestion or RAG benchmarks",
    )
    benchmark_sub = benchmark.add_subparsers(
        dest="benchmark_command",
        required=True,
    )
    ingestion = benchmark_sub.add_parser("ingestion")
    ingestion.add_argument("--dataset", required=True)
    ingestion.add_argument(
        "--output-root",
        default="artifacts/benchmarks/rag",
    )
    ingestion.add_argument("--config", default=None)

    retrieval = benchmark_sub.add_parser("retrieval")
    retrieval.add_argument("--dataset", required=True)
    retrieval.add_argument("--profiles", required=True)
    retrieval.add_argument("--profile-id", action="append", default=[])
    retrieval.add_argument(
        "--output-root",
        default="artifacts/benchmarks/rag",
    )
    retrieval.add_argument("--config", default=None)
    retrieval.add_argument(
        "--live",
        action="store_true",
        help="Allow model loading and isolated Qdrant/PostgreSQL access",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI grammar without executing a command."""
    parser = argparse.ArgumentParser(
        prog="litagent",
        description="LitAgent — Multi-agent adversarial literature review framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Survey command
    survey = subparsers.add_parser("survey", help="Run a full literature survey")
    survey.add_argument("query", help="Research query")
    survey.add_argument("--config", default=None, help="Path to config YAML")
    survey.add_argument("--output", "-o", default=None, help="Write report to file")
    survey.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    survey.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")

    # Configuration command
    cfg = subparsers.add_parser("config", help="Validate or display config")
    cfg.add_argument("--config", default=None, help="Path to config YAML")
    cfg.add_argument(
        "--validate-only",
        action="store_true",
        help="Silent validation: exit 0 if valid, exit 1 if invalid",
    )

    # Tool command
    tools_cmd = subparsers.add_parser("tools", help="List registered tools")
    tools_cmd.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )

    # Corpus command
    corpus = subparsers.add_parser(
        "corpus",
        help="Validate, ingest, inspect, or rebuild the paper corpus",
    )
    corpus_sub = corpus.add_subparsers(
        dest="corpus_command",
        required=True,
    )

    validate = corpus_sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)

    ingest = corpus_sub.add_parser("ingest")
    ingest.add_argument("--manifest", required=True)
    ingest.add_argument("--resume", action="store_true")

    stats = corpus_sub.add_parser("stats")
    stats.add_argument(
        "--purpose",
        choices=["runtime", "benchmark"],
        default="runtime",
    )

    rebuild = corpus_sub.add_parser("rebuild")
    rebuild.add_argument("--manifest", required=True)
    rebuild.add_argument("--yes", action="store_true")

    quarantine = corpus_sub.add_parser("quarantine")
    quarantine_sub = quarantine.add_subparsers(
        dest="quarantine_command",
        required=True,
    )
    quarantine_sub.add_parser("list")

    _add_benchmark_commands(subparsers)
    return parser


def main() -> None:
    """Parse arguments and dispatch the selected command."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "config":
        _cmd_config(args)
    else:
        asyncio.run(_dispatch_async(args))


async def _dispatch_async(args: argparse.Namespace) -> None:
    if args.command == "survey":
        await _cmd_survey(args)
    elif args.command == "tools":
        _cmd_tools(args)
    elif args.command == "corpus":
        await _cmd_corpus(args)
    elif args.command == "benchmark":
        await _cmd_benchmark(args)


async def _cmd_benchmark(args: argparse.Namespace) -> None:
    """Run one benchmark family and print its terminal summary."""
    import subprocess

    from litagent.benchmark.artifacts import BenchmarkArtifactRepository
    from litagent.benchmark.datasets import (
        load_ingestion_dataset,
        load_profiles,
        load_retrieval_dataset,
    )
    from litagent.benchmark.generated_ingestion import (
        GeneratedIngestionCaseExecutor,
    )
    from litagent.benchmark.ingestion_runner import IngestionRobustnessRunner
    from litagent.benchmark.rag_runner import RAGBenchmarkRunner

    config = load_config(args.config)
    artifacts = BenchmarkArtifactRepository(Path(args.output_root))
    if args.benchmark_command == "ingestion":
        dataset = load_ingestion_dataset(Path(args.dataset))
        result = await IngestionRobustnessRunner(
            GeneratedIngestionCaseExecutor(config.rag)
        ).run(dataset.cases)
        artifacts.write(result)
        print(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        if result.status != "succeeded":
            raise SystemExit(1)
        return

    if not args.live:
        raise SystemExit("retrieval benchmark requires explicit --live")
    dataset = load_retrieval_dataset(Path(args.dataset))
    profiles = load_profiles(Path(args.profiles))
    selected = set(args.profile_id)
    if selected:
        known = {profile.profile_id for profile in profiles}
        unknown = selected - known
        if unknown:
            raise SystemExit(f"unknown benchmark profiles: {sorted(unknown)}")
        profiles = [profile for profile in profiles if profile.profile_id in selected]
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    try:
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_dirty = True
    results = await RAGBenchmarkRunner(
        base_config=config,
        artifacts=artifacts,
    ).run(
        dataset=dataset,
        profiles=profiles,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )
    summary = {
        "status": (
            "succeeded"
            if all(result.status == "succeeded" for result in results)
            else "failed"
        ),
        "profiles": [
            {
                "profile_id": result.profile_id,
                "run_id": result.run_id,
                "status": result.status,
                "reason_codes": result.reason_codes,
            }
            for result in results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "succeeded":
        raise SystemExit(1)


async def _cmd_corpus(args: argparse.Namespace) -> None:
    """Dispatch corpus sub-commands (ingestion pipeline)."""
    import time

    from litagent.config import load_config
    from litagent.rag.chunking import build_corpus_chunker
    from litagent.rag.ingest import (
        CorpusIngestor,
        ParsedAuditRepository,
        QuarantineRepository,
    )
    from litagent.rag.manifest import load_manifest
    from litagent.rag.pdf_parser import PyMuPDFParser
    from litagent.rag.quality import CorpusTextQualityGate
    from litagent.rag.runtime import CorpusRuntime
    from litagent.rag.sources import ArxivPDFAdapter, LocalPDFAdapter
    from litagent.rag.vector_store import QdrantVectorStore

    config = load_config()
    started = time.monotonic()
    quarantine = QuarantineRepository(Path(config.rag.quarantine_root))

    if args.corpus_command == "quarantine":
        if args.quarantine_command == "list":
            entries = quarantine.list_entries()
            print(
                json.dumps(
                    {
                        "identity": config.rag.paper_collection,
                        "status": "ok",
                        "count": len(entries),
                        "entries": entries,
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

    if args.corpus_command == "validate":
        manifest = load_manifest(
            Path(args.manifest),
            raw_root=Path(config.rag.raw_root),
        )
        print(
            json.dumps(
                {
                    "identity": config.rag.paper_collection,
                    "status": "ok",
                    "counts": {"papers": len(manifest.papers)},
                    "reason_codes": [],
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    runtime = await CorpusRuntime.connect(
        config,
        purpose=getattr(args, "purpose", "runtime"),
    )
    try:
        if args.corpus_command == "stats":
            stats = await runtime.store.stats()
            print(
                json.dumps(
                    {
                        "identity": runtime.identity.collection_name,
                        "status": stats.status.value,
                        "counts": {"points": stats.points_count},
                        "reason_codes": [],
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        if args.corpus_command == "rebuild" and not args.yes:
            print(
                json.dumps(
                    {
                        "identity": runtime.identity.collection_name,
                        "status": "aborted",
                        "reason_codes": ["rebuild_requires_yes"],
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2)

        import httpx

        async with httpx.AsyncClient() as http_client:
            if args.corpus_command == "rebuild":
                # Rebuild the versioned collection before re-ingesting.
                if await runtime.qdrant_client.collection_exists(
                    runtime.identity.collection_name
                ):
                    await runtime.qdrant_client.delete_collection(
                        runtime.identity.collection_name
                    )
                dim = await asyncio.to_thread(lambda: runtime.embedder.dim)
                await QdrantVectorStore.ensure_compatible(
                    runtime.qdrant_client,
                    runtime.identity.collection_name,
                    dim,
                    identity=runtime.identity,
                    embedder=runtime.embedder,
                )
                await runtime.state.reset_collection(runtime.identity.collection_name)

            quality_gate = CorpusTextQualityGate(**config.rag.quality.model_dump())
            parser = PyMuPDFParser(
                quality_gate=quality_gate,
                chunker=build_corpus_chunker(config.rag),
            )
            local_pdf_adapter = LocalPDFAdapter(
                max_pdf_bytes=config.rag.max_pdf_bytes,
            )
            pdf_adapter = ArxivPDFAdapter(
                http_client,
                raw_root=Path(config.rag.raw_root),
                max_pdf_bytes=config.rag.max_pdf_bytes,
            )
            ingestor = CorpusIngestor(
                config=config.rag,
                service=runtime.service,
                parser=parser,
                local_pdf_adapter=local_pdf_adapter,
                pdf_adapter=pdf_adapter,
                quarantine=quarantine,
                audit_repository=ParsedAuditRepository(Path(config.rag.parsed_root)),
            )

            if args.corpus_command in {"ingest", "rebuild"}:
                summary = await ingestor.ingest_manifest(
                    Path(args.manifest),
                    resume=getattr(args, "resume", False),
                )
                outcome_counts: dict[str, int] = {}
                for report in summary.reports:
                    outcome = report.outcome.value
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                print(
                    json.dumps(
                        {
                            "identity": runtime.identity.collection_name,
                            "status": summary.status,
                            "counts": {
                                "papers": summary.paper_count,
                                "succeeded": summary.succeeded_count,
                                "failed": summary.failed_count,
                                "embedded": summary.embedded_count,
                                "payload_updated": summary.payload_updated_count,
                                "deleted": summary.deleted_count,
                                "unchanged": summary.unchanged_count,
                            },
                            "outcomes": outcome_counts,
                            "reports": [
                                report.model_dump(mode="json")
                                for report in summary.reports
                            ],
                            "reason_codes": summary.reason_codes,
                            "elapsed_ms": int((time.monotonic() - started) * 1000),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
    finally:
        await runtime.close()


def _delivery_exit_code(report: dict) -> int:
    """Return zero only when the report is publishable."""
    delivery = report.get("delivery") or derive_delivery(
        report.get("partial", False), report.get("quality")
    )
    return 0 if delivery.get("publishable", False) else 1


async def _cmd_survey(args: argparse.Namespace) -> None:
    """Run a survey and emit it in the requested format."""
    config = load_config(args.config)
    if args.verbose:
        config.logging.level = "DEBUG"

    async with LitAgent(config) as agent:
        report = await agent.run(args.query)

    if args.format == "json":
        output = json.dumps(report, ensure_ascii=False, indent=2)
    else:
        output = _format_report_markdown(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)

    code = _delivery_exit_code(report)
    if code:
        delivery = report.get("delivery") or derive_delivery(
            report.get("partial", False), report.get("quality")
        )
        print(
            f"\nDelivery: {delivery.get('status', 'unknown')} — not publishable",
            file=sys.stderr,
        )
        sys.exit(code)


def _cmd_config(args: argparse.Namespace) -> None:
    """Validate or display the resolved configuration."""
    try:
        config = load_config(args.config)
    except Exception as e:
        if args.validate_only:
            print(f"INVALID: {e}", file=sys.stderr)
            sys.exit(1)
        raise

    if args.validate_only:
        print("OK: config is valid")
        sys.exit(0)

    print(config.model_dump_json(indent=2))


def _cmd_tools(args: argparse.Namespace) -> None:
    """List the registered built-in tools."""
    from litagent.tools.builtin.extract import register_extract_tools
    from litagent.tools.builtin.search import register_search_tools
    from litagent.tools.registry import get_registry

    registry = get_registry()
    register_search_tools()
    register_extract_tools()
    tools = registry.list_all()

    if args.format == "json":
        data = [
            {"name": t.name, "category": t.category.value, "description": t.description}
            for t in tools
        ]
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"{'Name':<35} {'Category':<12} {'Description':<50}")
        print("-" * 97)
        for t in sorted(tools, key=lambda t: t.name):
            desc = (
                t.description[:47] + "..." if len(t.description) > 50 else t.description
            )
            print(f"{t.name:<35} {t.category.value:<12} {desc:<50}")


def _format_report_markdown(report: dict) -> str:
    """Render a survey report as Markdown."""
    survey_text = report.get("survey", "")
    metadata = report.get("metadata", {})
    review_history = report.get("review_history", [])

    lines = [
        "# Literature Survey Report",
        "",
        f"**Query**: {metadata.get('query', 'N/A')}",
        f"**Generated**: {metadata.get('generated_at', 'N/A')}",
        f"**Review rounds**: {metadata.get('total_rounds', 0)}",
        f"**Final score**: {metadata.get('final_score', 'N/A')}",
        f"**Accepted**: {metadata.get('accepted', False)}",
    ]

    delivery = report.get("delivery") or derive_delivery(
        report.get("partial", False), report.get("quality")
    )
    quality = report.get("quality") or {}

    banner = None
    if delivery["status"] == "partial":
        banner = (
            "> ⚠ **Partial results — NOT PUBLISHABLE** — "
            "survey was interrupted (cost/timeout)."
        )
    elif delivery["status"] == "blocked":
        banner = (
            "> ⚠ **QUALITY FAILED — UNTRUSTED DRAFT, NOT PUBLISHABLE** "
            f"(failed: {', '.join(quality.get('failed_metrics', []))})"
        )
    elif delivery["status"] == "needs_review":
        banner = "> ⚠ **Quality unverified — needs review before publishing**"
    if banner:
        lines.insert(0, banner)
        lines.insert(0, "")

    lines.extend(["", "---", "", survey_text, ""])

    if review_history:
        lines.append("## Review History")
        lines.append("")
        for r in review_history:
            review = r.get("review", {})
            lines.append(
                f"### Round {r.get('round', '?')} — "
                f"Score: {review.get('score', 'N/A')}, "
                f"Verdict: {review.get('verdict', 'N/A')}"
            )
            for w in review.get("weaknesses", []):
                lines.append(f"- ⚠ {w}")
            for issue in review.get("issues", []):
                lines.append(
                    f"- [{issue.get('severity', 'minor')}] "
                    f"{issue.get('section', '')}: {issue.get('issue', '')}"
                )
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
