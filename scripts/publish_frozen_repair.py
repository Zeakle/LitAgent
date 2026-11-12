"""Publish the frozen few-shot repair case from its settled local archives."""

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

from litagent.quality import digest

STEM = "frozen-repair-377-380"
JOURNAL = "repair.json.writer-v2.json"
CITATION = re.compile(r"\[E:([^\]]+)\]")
PAPER = re.compile(r"arxiv:(\d{4}\.\d{4,5}(?:v\d+)?)")


def _sha(data):
    """Hash either exact bytes or the UTF-8 representation of Markdown."""
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _read(path):
    """Read a local JSON witness without modifying it."""
    return json.loads(path.read_bytes())


def _require(condition, reason):
    """Reject identity or source mismatches before publishing an excerpt."""
    if not condition:
        raise ValueError(reason)


def _candidate(root, *, publishable):
    """Load one real candidate and verify the terminal receipt and new usage."""
    receipt = _read(root / "live-exit.json")
    manifest = _read(root / "manifest.json")
    _require(
        receipt["actual_exit_code"] == 0
        and receipt["source_files_verified"]
        and receipt["publishable"] is publishable
        and receipt["manifest_identity"] == manifest["identity"]
        and all(
            receipt[key] == _sha((root / name).read_bytes())
            for key, name in (
                ("candidate_journal_sha256", JOURNAL),
                ("ledger_sha256", "ledger.json"),
                ("result_sha256", "result.json"),
            )
        ),
        "terminal_receipt_mismatch",
    )
    (candidate,) = _read(root / JOURNAL)["candidates"]
    result = _read(root / "result.json")
    ledger = _read(root / "ledger.json")
    sha = _sha(candidate["survey"])
    details = candidate["evaluation"]["faithfulness"]["details"]
    _require(
        sha == candidate["report_sha256"] == details["source_report_sha256"]
        and details["evidence_snapshot_sha256"] == manifest["evidence_snapshot_sha256"]
        and candidate["committed"] is result["candidate_committed"] is publishable
        and result["acceptance"]["publishable"] is publishable
        and not ledger["usage_uncertain"]
        and result["usage_uncertain"] is False
        and result["new_unknown_hold"] == 0
        and result["new_requests"] == receipt["new_requests"]
        and result["new_known_tokens"] == receipt["new_known_tokens"],
        "candidate_or_usage_mismatch",
    )
    return candidate, details, result, manifest, receipt


def _counts(candidate, details, result):
    """Keep observed verdict counts distinct from the formal metric."""
    counts = Counter(
        row["status"] for row in details["assertions"] if row["kind"] == "scientific"
    )
    _require(
        counts.total() == details["factual_claim_count"]
        and all(
            counts[key] == details[key + "_count"]
            for key in ("supported", "unsupported", "unverified")
        ),
        "scientific_counts_mismatch",
    )
    return {
        "report_sha256": candidate["report_sha256"],
        "scientific": counts.total(),
        **{key: counts[key] for key in ("supported", "unsupported", "unverified")},
        "faithfulness": candidate["evaluation"]["faithfulness"]["score"],
        "inventory": details["inventory_check"]["status"],
        "global_check": details["stages"]["global_check"],
        "global_input_contract": details["stages"].get("global_input_contract"),
        "publishable": result["acceptance"]["publishable"],
        "requests": result["new_requests"],
        "tokens": result["new_known_tokens"],
        "unknown_hold": result["new_unknown_hold"],
    }


def _unit(candidate, details, unit_id, evidence):
    """Bind a public excerpt and its claims to exact offsets and cited sources."""
    (unit,) = (
        row for row in candidate["bindings"]["units"] if row["unit_id"] == unit_id
    )
    start, end = unit["source_start"], unit["source_end"]
    original = candidate["survey"][start:end]
    cited = unit["evidence_ids"]
    _require(
        original == unit["source_quote"]
        and cited
        and CITATION.findall(original) == cited
        and all(key in evidence for key in cited),
        "excerpt_source_mismatch",
    )
    rows = [
        row
        for row in details["assertions"]
        if (row["source_start"], row["source_end"]) == (start, end)
    ]
    _require(
        rows
        and all(
            row["source_quote"] == original and row["evidence_ids"] == cited
            for row in rows
        ),
        "assertion_source_mismatch",
    )
    return {
        "start": start,
        "end": end,
        "excerpt": CITATION.sub("", original).rstrip(),
        "evidence_ids": cited,
        "assertions": [
            {
                "id": row["claim_id"],
                "text": row["claim_text"],
                "verdict": row["status"],
            }
            for row in rows
            if row["kind"] == "scientific"
        ],
    }


def publication(before: Path, after: Path) -> dict:
    """Project only verified excerpts, verdicts and public paper locators."""
    old, old_details, old_result, old_manifest, old_exit = _candidate(
        before, publishable=False
    )
    new, new_details, new_result, new_manifest, new_exit = _candidate(
        after, publishable=True
    )
    report = _read(after / "report.json")
    fact_cards = report["metadata"]["fact_cards"]
    evidence = {card["evidence_id"]: card for card in fact_cards.values()}
    rounds = _read(after / "rounds.json")["rounds"]
    (completed,) = rounds
    _require(
        new_manifest["parent_report_sha256"] == old["report_sha256"]
        and new["source_report_sha256"]
        == completed["source_report_sha256"]
        == old["report_sha256"]
        and new["report_sha256"] == completed["candidate_sha256"]
        and completed["committed"] is True
        and new_exit["rounds_sha256"] == _sha((after / "rounds.json").read_bytes())
        and new_exit["publishable_sha256"]
        == _sha((after / "publishable.md").read_bytes())
        == new_result["acceptance"]["report_sha256"]
        == new["report_sha256"]
        and new["survey"] == report["survey"]
        and new["evaluation"] == report["evaluation"]
        and digest(report["evaluation"])
        == new_result["acceptance"]["evaluation_sha256"]
        and digest(fact_cards)
        == old_manifest["evidence_snapshot_sha256"]
        == new_manifest["evidence_snapshot_sha256"]
        == new_result["acceptance"]["fact_cards_sha256"]
        and old_manifest["models"] == new_manifest["models"],
        "repair_lineage_mismatch",
    )
    old_row = _counts(old, old_details, old_result)
    new_row = _counts(new, new_details, new_result)
    _require(
        old_row["faithfulness"] is None
        and new_row["faithfulness"] == 1.0
        and new_row["supported"] == new_row["scientific"]
        and new_details["stages"]["global_check"] == "completed"
        and not new_details["global_issues"],
        "evaluation_state_mismatch",
    )
    repairs = []
    evidence_ids = set()
    for unit_id in completed["decision"]["targets"]:
        original = _unit(old, old_details, unit_id, evidence)
        revised = _unit(new, new_details, unit_id, evidence)
        rejected = [
            row for row in original["assertions"] if row["verdict"] == "unsupported"
        ]
        _require(rejected and revised["assertions"], "repair_target_mismatch")
        original["assertions"] = rejected
        evidence_ids.update(original["evidence_ids"] + revised["evidence_ids"])
        repairs.append({"unit_id": unit_id, "before": original, "after": revised})
    _require(
        len(repairs) == new_manifest["authorized_target_count"]
        and len({row["unit_id"] for row in repairs}) == len(repairs)
        and {claim["id"] for row in repairs for claim in row["before"]["assertions"]}
        == {row["claim_id"] for row in old_details["unsupported_claims"]},
        "repair_coverage_mismatch",
    )
    sources = {}
    for key in sorted(evidence_ids):
        card = evidence[key]
        paper = PAPER.fullmatch(card["paper_id"])
        _require(paper is not None, "public_paper_locator_invalid")
        sources[key] = {
            "paper_id": card["paper_id"],
            "url": f"https://arxiv.org/abs/{paper[1]}",
            "content_sha256": card["content_sha256"],
        }
    return {
        "id": STEM,
        "contract": "frozen-repair-public-v1",
        "scope": "single_few_shot_frozen_evidence_revision",
        "human_accuracy": None,
        "formal_benchmark": False,
        "models": new_manifest["models"],
        "frozen_evidence_count": len(fact_cards),
        "evidence_snapshot_sha256": new_manifest["evidence_snapshot_sha256"],
        "lineage_prose_repairs": new_result["lineage_prose_repairs"],
        "before": old_row,
        "after": new_row,
        "before_global_findings": [
            {
                "code": row["code"],
                "unit_id": row["source_unit_id"],
                "recorded_assertions": sum(
                    claim["source_unit_id"] == row["source_unit_id"]
                    for claim in old_details["assertions"]
                ),
            }
            for row in old_details["global_issues"]
        ],
        "checks": {
            "core_covered": sum(
                row["answered"] for row in new_details["core_coverage"]
            ),
            "core_total": len(new_details["core_coverage"]),
            "analysis": new_details["analysis_check"]["status"],
            "citation_id_validity": new["evaluation"]["citation_accuracy"]["score"],
            "internal_consistency": new["evaluation"]["internal_consistency"]["score"],
        },
        "receipts": {
            "377_candidate_journal_sha256": old_exit["candidate_journal_sha256"],
            "380_candidate_journal_sha256": new_exit["candidate_journal_sha256"],
            "377_terminal_sha256": _sha((before / "live-exit.json").read_bytes()),
            "380_terminal_sha256": _sha((after / "live-exit.json").read_bytes()),
            "380_rounds_sha256": new_exit["rounds_sha256"],
        },
        "repairs": repairs,
        "sources": sources,
    }


def render_html(data: dict) -> str:
    """Render the public case with native links and disclosures."""
    esc = html.escape
    before, after = data["before"], data["after"]
    repairs, sources = data["repairs"], data["sources"]
    source_numbers = {key: i for i, key in enumerate(sources, 1)}
    checks = data["checks"]
    claim_index = []
    sections = []
    for i, repair in enumerate(repairs, 1):
        columns = []
        for version, label in (
            ("before", "377 · 修订前 / Before"),
            ("after", "380 · 修订后 / After"),
        ):
            unit = repair[version]
            links = " · ".join(
                f'<a href="#evidence-{source_numbers[key]}">E{source_numbers[key]}</a>'
                for key in unit["evidence_ids"]
            )
            claims = []
            for claim in unit["assertions"]:
                anchor = f"{version}-{esc(claim['id'])}"
                claims.append(
                    f'<li id="{anchor}"><code>{esc(claim["id"])}</code> '
                    f'{esc(claim["text"])} <small>({esc(claim["verdict"])})</small></li>'
                )
                if version == "before":
                    claim_index.append(f'<a href="#{anchor}">{esc(claim["id"])}</a>')
            claim_list = f'<ul>{"".join(claims)}</ul>'
            if version == "after":
                claim_list = f"<details><summary>修订后抽取的断言 / Revised assertions</summary>{claim_list}</details>"
            columns.append(
                f"<div><h4>{label}</h4>{claim_list}"
                f'<p><a href="#{version}-unit-{i}">定位报告原句 / Locate passage</a> · {links}</p>'
                f'<blockquote id="{version}-unit-{i}">{esc(unit["excerpt"])}</blockquote>'
                f'<p class="meta">字符范围 / Character offsets [{unit["start"]}, {unit["end"]})</p></div>'
            )
        sections.append(
            f'<article><h3>{i}. <code>{esc(repair["unit_id"])}</code></h3>'
            f'<div class="compare">{"".join(columns)}</div></article>'
        )
    source_rows = "".join(
        f'<li id="evidence-{i}">E{i} · <a href="{esc(source["url"])}">{esc(source["paper_id"])}</a>'
        f"<br><code>{esc(key)}</code></li>"
        for i, (key, source) in enumerate(sources.items(), 1)
    )
    global_rows = " · ".join(
        f'<code>{esc(row["code"])}</code> <code>{esc(row["unit_id"])}</code>'
        f'（该单元已有 {row["recorded_assertions"]} 条断言）'
        for row in data["before_global_findings"]
    )
    table_rows = "".join(
        f'<tr><th scope="row">{label}</th><td>{esc(str(left))}</td><td>{esc(str(right))}</td></tr>'
        for label, left, right in (
            (
                "逐条支持 / Supported",
                f'{before["supported"]}/{before["scientific"]}',
                f'{after["supported"]}/{after["scientific"]}',
            ),
            ("不支持 / Unsupported", before["unsupported"], after["unsupported"]),
            ("未核验 / Unverified", before["unverified"], after["unverified"]),
            (
                "正式 Faithfulness",
                "不可用 / Unavailable",
                f'{after["faithfulness"]:.2f}',
            ),
            ("严格交付 / Delivery", "拒绝 / Blocked", "通过 / Passed"),
            ("本轮请求 / Requests", before["requests"], after["requests"]),
            ("本轮已结算 tokens", f'{before["tokens"]:,}', f'{after["tokens"]:,}'),
            ("未知占用 / Unknown hold", before["unknown_hold"], after["unknown_hold"]),
        )
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>377 → 380 · LitAgent Benchmark</title>
<style>
body{{font:16px/1.65 system-ui;margin:24px auto;padding:0 16px;max-width:1100px;color:#17263a}}
a{{color:#164b9b}} table{{border-collapse:collapse;width:100%}}
td,th{{padding:.55em;border-bottom:1px solid #b9c9dc;text-align:left}}
article{{border-top:1px solid #b9c9dc;margin-top:1.25em}}
code,li,td,th{{overflow-wrap:anywhere}} .meta{{font-size:.9em}}
.compare{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1em}}
blockquote{{margin:.6em 0;padding:.75em;background:#edf3fa}}
:target{{outline:2px solid #164b9b;outline-offset:2px}}
@media(max-width:680px){{.compare{{grid-template-columns:1fr}}td,th{{padding:.35em}}}}
</style></head>
<body><main>
<nav><a href="/presentation">公开演示 / Presentation</a> · <a href="/flow-demo?view=benchmark">工作台 Benchmark</a></nav>
<h1>377 → 380 · 冻结证据修订案例</h1>
<p lang="en">Frozen-evidence repair of one few-shot learning survey.</p>
<p>真实付费模型运行，复用 {data["frozen_evidence_count"]} 条冻结证据。本案例仅验证独立修订与完整复评。
正式 15 题套件单列，人工校准状态为未验证。Each version has its own assertion inventory.</p>
<table><caption>两个候选各自的 Evaluation 与实际用量</caption>
<thead><tr><th scope="col">指标 / Metric</th><th scope="col">377</th><th scope="col">380</th></tr></thead>
<tbody>{table_rows}</tbody></table>
<p>377 的逐条核验已返回，清单因全局遗漏发现未通过，正式支持率不可用。全局记录为 {global_rows}。</p>
<p>380 从 377 候选出发再修一轮，更新摘要与结论后重做全篇 Evaluation。核心覆盖
{checks["core_covered"]}/{checks["core_total"]}，章节分析通过，引用 ID 有效性与内部一致性均为 1.00。
全局检查完成且无问题。达到严格交付后停止。</p>
<p>新全局输入将同一原文单元与其断言放在一起
<code>{esc(after["global_input_contract"])}</code>。前后两版保留各自的原始判定。</p>
<p>正文修订 <code>{esc(data["models"]["generation"])}</code> ·
评估 <code>{esc(data["models"]["evaluation"])}</code></p>
<details><summary>版本与冻结证据 SHA-256 / Version bindings</summary><ul>
<li>377 <code>{esc(before["report_sha256"])}</code></li>
<li>380 <code>{esc(after["report_sha256"])}</code></li>
<li>Evidence <code>{esc(data["evidence_snapshot_sha256"])}</code></li></ul></details>
<h2>从断言到原句 / Trace the rejected assertions</h2>
<p>九条负项对应八个授权原文单元。对照以原文单元为准，两版断言分别抽取。点选断言，再查看该段实际引用。</p>
<p>{" · ".join(claim_index)}</p>
{"".join(sections)}
<h2>实际引用的公开论文 / Cited papers</h2><ol>{source_rows}</ol>
<p><a href="{STEM}.json">同源数据与回执 SHA / Data and receipt hashes</a></p>
</main></body></html>
"""


def main():
    """Generate both public formats from the same verified projection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = publication(args.before, args.after)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{STEM}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / f"{STEM}.html").write_text(render_html(data), encoding="utf-8")


if __name__ == "__main__":
    main()
