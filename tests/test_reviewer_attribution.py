"""Require grounded Reviewer attribution before granting prose-edit permission."""

import json
from pathlib import Path

import pytest
from test_separate_review_pipeline import execute
from test_stream_writer_pipeline import StreamingGateway, stream_config
from test_writer_v2 import case  # noqa: F401

from litagent import stage_repair
from litagent.llm.responses import decode_private_receipt


def test_vit14_reviewer_wrong_paper_cannot_authorize_either_edit():
    root = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-14"
    )
    journal_path = root / "repairs.json.writer-v2.json"
    report_path = root / "runs/vit-235d5bf1dba2-inline-reassessment-14.json"
    if not journal_path.exists() or not report_path.exists():
        pytest.skip("requires the immutable local ViT14 receipts")
    journal_raw, report_raw = journal_path.read_bytes(), report_path.read_bytes()
    call = json.loads(journal_raw)["calls"]["reviewer:0"]
    request, _ = decode_private_receipt(call["provider_receipt"])
    payload = json.loads(request["input"][1]["content"])
    metadata = json.loads(report_raw)["report"]["metadata"]
    doc = metadata["writer_document"]
    assert metadata["writing_bindings"]["report_sha256"] == payload["report_sha256"]
    aliases = payload["citation_projection"]
    assert doc["packet"]["cards"]["c25"]["evidence_id"] == aliases["c25"]
    bad = []
    for old in json.loads(call["raw"])["issues"]:
        assert "ED-ViT" in old["explanation"]
        text = payload["units"][old["unit_id"]]["source_quote"]
        end = text.index("[E:c25]") + len("[E:c25]")
        bad.append(
            {
                **old,
                "kind": "citation",
                "card_ref": "c25",
                "report_excerpt": text[end - 16 : end],
                "source_excerpt": "we propose a novel Vision Transformer splitting framework, ED-ViT",
            }
        )
    src = doc["packet"]["sources"]
    assert (
        bad[0]["source_excerpt"]
        in src[doc["packet"]["cards"]["c19"]["source_ref"]]["source_quote"]
    )
    args = (doc["packet"], aliases, metadata["writing_bindings"]["citation_scopes"])
    eligible, rejected = stage_repair.bind_reviewer_issues(bad, *args)
    assert eligible == []
    assert [issue["code"] for issue in rejected] == [
        "reviewer_source_excerpt_unbound"
    ] * 2

    # A literally bound source is a valid basis for review, not a truth verdict.
    example = {
        **bad[0],
        "source_excerpt": "our small ViTSTR achieves a competi-tive accuracy",
    }
    eligible, rejected = stage_repair.bind_reviewer_issues([bad[1], example], *args)
    assert eligible == [
        {"unit_id": example["unit_id"], "explanation": example["explanation"]}
    ]
    assert len(rejected) == 1
    assert (
        journal_path.read_bytes() == journal_raw
        and report_path.read_bytes() == report_raw
    )


@pytest.mark.parametrize(
    ("report_excerpt", "source_excerpt", "code"),
    [
        (
            "First claim[E:c1]。Second claim",
            "Exact source passage",
            "reviewer_report_citation_unbound",
        ),
        ("[E:c1]。", "Exact source passage", "reviewer_report_citation_unbound"),
        ("First claim[E:c1]", ".", "reviewer_source_excerpt_unbound"),
    ],
)
def test_reviewer_excerpt_must_bind_report_prose_and_source_text(
    report_excerpt, source_excerpt, code
):
    aliases = {"c1": "paper-a", "c2": "paper-b"}
    packet = {
        "cards": {"c1": {"source_ref": "a"}, "c2": {"source_ref": "b"}},
        "sources": {"a": {"source_quote": "Exact source passage."}},
    }
    scopes = [
        {
            "unit_id": "u",
            "source_quote": "First claim[E:paper-a]。",
            "evidence_ids": ["paper-a"],
        },
        {
            "unit_id": "u",
            "source_quote": "Second claim[E:paper-b]。",
            "evidence_ids": ["paper-b"],
        },
    ]
    wrong = {
        "unit_id": "u",
        "explanation": "An unrelated source.",
        "kind": "citation",
        "card_ref": "c1",
        "source_excerpt": source_excerpt,
        "report_excerpt": report_excerpt,
    }
    eligible, rejected = stage_repair.bind_reviewer_issues(
        [wrong], packet, aliases, scopes
    )
    assert eligible == []
    assert rejected[0]["code"] == code


@pytest.mark.asyncio
@pytest.mark.parametrize("with_valid_issue", [False, True], ids=["unbound", "mixed"])
async def test_unbound_reviewer_attribution_preserves_independent_evaluation(
    case, tmp_path, monkeypatch, with_valid_issue
):
    class Gateway(StreamingGateway):
        allowed = None
        editorial_targets = None

        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            fields = schema.get("properties", {})
            if "target_responses" in fields and self.global_calls == 0:
                self.editorial_targets = set(payload["response_target_keys"].values())
            if "issues" in fields and self.review_calls == 1:
                props = schema["properties"]["issues"]["items"]["properties"]
                uid, quote = next(
                    (uid, unit["source_quote"])
                    for uid, unit in payload["units"].items()
                    if ":methods:" in uid and "[E:c1]" in unit["source_quote"]
                )
                end = quote.index("[E:c1]") + len("[E:c1]")
                assert "c1" in props["card_ref"]["enum"]
                result["issues"] = [
                    {
                        "kind": "citation",
                        "unit_id": uid,
                        "card_ref": "c1",
                        "report_excerpt": quote[end - 12 : end],
                        "source_excerpt": "This literal does not occur in the original.",
                        "explanation": "Unsupported attribution concern.",
                    }
                ]
                if with_valid_issue:
                    self.allowed = next(
                        uid for uid in payload["units"] if ":introduction:" in uid
                    )
                    result["issues"].append(
                        {
                            "kind": "editorial",
                            "unit_id": self.allowed,
                            "card_ref": "",
                            "report_excerpt": "",
                            "source_excerpt": "",
                            "explanation": "Clarify the review scope.",
                        }
                    )
            return result

    gateway = Gateway(case)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stream_config())
    assert gateway.review_calls == (2 if with_valid_issue else 1)
    assert gateway.editorial_edits == int(with_valid_issue)
    assert gateway.editorial_targets == (
        {gateway.allowed} if with_valid_issue else None
    )
    assert (
        report["evaluation"]["faithfulness"]["details"]["stages"]["global_check"]
        == "completed"
    )
    recorded = report["metadata"]["editorial_history"][0]
    assert recorded["status"] == (
        "revised_unverified" if with_valid_issue else "review_issues_unbound"
    )
    assert recorded["review"]["accepted"] is False
    assert (
        recorded["review"]["rejected_issues"][0]["code"]
        == "reviewer_source_excerpt_unbound"
    )
