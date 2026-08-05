"""Run survey, configuration, and tool-listing commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from litagent.config import load_config
from litagent.runner import LitAgent, derive_delivery


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


async def _cmd_corpus(args: argparse.Namespace) -> None:
    """Dispatch corpus sub-commands (ingestion pipeline)."""
    import time

    from litagent.config import load_config
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
            parser = PyMuPDFParser(quality_gate=quality_gate)
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
