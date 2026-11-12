"""Audit necessary reservations without creating a runtime or admitting a plan."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

from litagent.context.evidence_selector import EvidenceSelection
from litagent.frozen_writing import CASES, read_bundle
from litagent.quality import build_fact_cards, digest
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import complete_writer_plan
from litagent.writing_v2 import source_packet

CONTRACT = "writer-reservation-floor-v1"


def reservation_floor(packet: dict) -> dict:
    """Exclude impossible reservations; a lower bound can never authorize sending.

    The legal maximum path reserves all eight assessment batches for the initial
    report and both repair candidates. Five chapter calls and two body patches
    each include the complete packet. Summaries do not include that packet.
    All other inputs are deliberately omitted, making this a necessary portion
    of the existing conservative reservation, not a lower bound on billed tokens.
    """
    if (
        not isinstance(packet, dict)
        or not packet.get("sources")
        or not packet.get("cards")
    ):
        raise ValueError("reservation_floor_packet_invalid")
    serialized = json.dumps(packet, ensure_ascii=False).encode()
    plan = complete_writer_plan(identity=digest(packet))
    stages = plan["stages"]
    count = sum(s["count"] for s in stages.values())
    outputs = sum(s["count"] * s["output_tokens"] for s in stages.values())
    # Measure the existing counter rather than inventing another padding rule.
    padding = strict_request_reservation([], 0) - len(json.dumps([]).encode())
    copies = stages["body"]["count"] + stages["repair"]["count"]
    # Members survive unchanged inside chapter/patch payloads. Braces are not
    # counted; outer message escaping and all extra fields only add bytes.
    packet_members = len(serialized) - 2
    required = outputs + count * padding + copies * packet_members
    allowance = max(500000, ((required * 11 + 99999) // 100000) * 10000)
    return {
        "contract": CONTRACT,
        "measurement": plan["measurement"],
        "packet_sha256": digest(packet),
        "path_requests": count,
        "output_reservations": outputs,
        "padding_per_request": padding,
        "padding_reservations": count * padding,
        "packet_member_bytes": packet_members,
        "packet_copies": copies,
        "packet_reservations": copies * packet_members,
        "necessary_reservations": required,
        "minimum_computed_allowance": allowance,
        "maximum_allowance": plan["maximum_tokens"],
        "ceiling_conflict_proven": allowance > plan["maximum_tokens"],
        "can_admit": False,
        "evaluation_completed": False,
        "new_paid_requests": 0,
        "omitted_positive_inputs": [
            "system instructions and native schemas",
            "prior chapters and current repair bodies",
            "summary and reviewer inputs",
            "all evaluation inputs, including original support passages",
            "diagnostics, request envelopes and outer JSON escaping",
        ],
    }


def build_review() -> dict:
    """Validate the original three snapshots and export only safe scalar facts."""
    rows = []
    for case_id, archive in CASES:
        bundle = read_bundle(Path("artifacts/runs", archive + ".json"))
        selection = EvidenceSelection.from_dict(bundle["selection"])
        cards = build_fact_cards(selection.selected_items)
        if digest(cards) != bundle["fact_cards_sha256"]:
            raise ValueError("frozen_writing_fact_card_identity_conflict")
        packet, _ = source_packet(
            cards, selection, bundle["query"], bundle["requirements"], "zh-CN"
        )
        rows.append(
            {
                "case_id": case_id,
                "archive_sha256": bundle["archive_sha256"],
                **reservation_floor(packet),
            }
        )
    result = {
        "contract": CONTRACT,
        "kind": "offline-reservation-feasibility-not-scientific-benchmark",
        "new_paid_requests": 0,
        "cases": rows,
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve().relative_to(Path.cwd()),
                Path("src/litagent/safety/planned_budget.py"),
                Path("src/litagent/safety/budget.py"),
                Path("src/litagent/writing_v2.py"),
                Path("src/litagent/frozen_writing.py"),
            )
        },
    }
    result["identity"] = digest(result)
    return result


def export_review(review: dict, directory: Path) -> None:
    """Render identical scalar diagnostics; never publish prompts or source text."""
    directory.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(review, ensure_ascii=False, indent=2)
    (directory / "review.json").write_text(serialized + "\n", encoding="utf-8")
    fields = [
        "case_id",
        "path_requests",
        "output_reservations",
        "padding_reservations",
        "packet_reservations",
        "necessary_reservations",
        "minimum_computed_allowance",
        "ceiling_conflict_proven",
        "can_admit",
        "new_paid_requests",
    ]
    with (directory / "review.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review["cases"])
    explanation = (
        "离线审查，新增付费请求为 0。这是当前保守计量的必要部分预留下界，"
        "不是实际费用，也不是实际模型完成流程所需 tokens 的最低值。"
        "完整上界才可用于准入；下界未超限也不能启动。"
        "41 次请求包括五章、三次摘要、一次 Reviewer、两次正文修复，"
        "以及三次完整 Evaluation（每次抽取＋八批支持核验＋全局检查）。"
        "保留 10% 余量后，当前三题的必要部分已超过 100 万。"
        "尚未计入的系统提示、schema、正文、诊断和所有 Evaluation 输入只会增加预留。"
        "正文质量尚未验证，Phase 14.5 仍为 in_progress。"
    )
    heading = "完整流程预算：必要部分的可行性审查"
    labels = [
        "题目",
        "请求数",
        "输出预留",
        "包装预留",
        "冻结材料部分",
        "必要部分合计",
        "加余量后的下界",
        "已证明超限",
        "允许准入",
        "新增付费请求",
    ]
    table = "|" + "|".join(fields) + "|\n|" + "|".join(["---"] * len(fields)) + "|\n"
    table += "\n".join(
        "|" + "|".join(str(row[k]) for k in fields) + "|" for row in review["cases"]
    )
    (directory / "review.md").write_text(
        f"# {heading}\n\n{explanation}\n\n{table}\n", encoding="utf-8"
    )
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in fields)
        + "</tr>"
        for row in review["cases"]
    )
    (directory / "review.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{heading}</title><style>body{{font:17px/1.7 system-ui;"
        "max-width:1100px;margin:32px auto;padding:0 20px;color:#172533}"
        "table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccd5df}"
        ".scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        f"</style><h1>{heading}</h1><p>{explanation}</p>"
        '<p><a href="review.json">JSON</a> · <a href="review.csv">CSV</a> · '
        '<a href="review.md">Markdown</a></p><div class="scroll" tabindex="0" '
        'role="region" aria-label="三域必要预留，可横向滚动"><table><thead><tr>'
        + "".join(f"<th scope=col>{key}</th>" for key in labels)
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div><details><summary>同源数据与来源指纹</summary><pre>"
        + html.escape(serialized)
        + '</pre></details><script id="data" type="application/json">'
        + serialized.replace("<", "\\u003c")
        + "</script></html>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    value = build_review()
    export_review(
        value, Path("benchmarks/survey/results/writer-budget-feasibility-20261027")
    )
    print(json.dumps({"new_paid_requests": 0, "cases": len(value["cases"])}))
