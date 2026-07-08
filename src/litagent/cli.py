# src/litagent/cli.py
"""CLI entry point for LitAgent.

Commands:
    litagent survey <query>    Run a full literature survey
    litagent config            Validate or display config
    litagent tools             List registered tools
"""

from __future__ import annotations
import argparse
import asyncio
import json
import sys

from litagent.config import load_config
from litagent.runner import LitAgent


def main() -> None:
    """CLI 主入口。pyproject.toml 中 [project.scripts] 指向此函数。"""
    parser = argparse.ArgumentParser(
        prog="litagent",
        description="LitAgent — Multi-agent adversarial literature review framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ── survey ──
    survey = subparsers.add_parser("survey", help="Run a full literature survey")
    survey.add_argument("query", help="Research query")
    survey.add_argument("--config", default=None, help="Path to config YAML")
    survey.add_argument("--output", "-o", default=None, help="Write report to file")
    survey.add_argument(
        "--format", choices=["json", "markdown"], default="markdown",
        help="Output format (default: markdown)",
    )
    survey.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")

    # ── config ──
    cfg = subparsers.add_parser("config", help="Validate or display config")
    cfg.add_argument("--config", default=None, help="Path to config YAML")
    cfg.add_argument(
        "--validate-only", action="store_true",
        help="Silent validation: exit 0 if valid, exit 1 if invalid",
    )

    # ── tools ──
    tools_cmd = subparsers.add_parser("tools", help="List registered tools")
    tools_cmd.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="Output format",
    )

    args = parser.parse_args()

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


async def _cmd_survey(args: argparse.Namespace) -> None:
    """运行完整 survey → 格式化输出。"""
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


def _cmd_config(args: argparse.Namespace) -> None:
    """校验或展示配置。"""
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
    """列出所有已注册 tool。"""
    from litagent.tools.registry import get_registry
    from litagent.tools.builtin.search import register_search_tools
    from litagent.tools.builtin.extract import register_extract_tools

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
            desc = t.description[:47] + "..." if len(t.description) > 50 else t.description
            print(f"{t.name:<35} {t.category.value:<12} {desc:<50}")


def _format_report_markdown(report: dict) -> str:
    """将 report dict 格式化为 Markdown。"""
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

    if report.get("partial"):
        lines.insert(1, "> ⚠ **Partial results** — survey was interrupted (cost/timeout).")
        lines.insert(1, "")

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