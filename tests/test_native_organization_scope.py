"""Keep scientific content out of the host organization check."""

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.eval.engine_policy import (
    ARCHIVED_INLINE_BOUND_ORGANIZATION_ENGINE,
    ARCHIVED_INLINE_PRE_ORGANIZATION_ENGINE,
    BOUND_ORGANIZATION,
    configured_engine,
    validate_engine,
)
from litagent.eval.ragas_native import canonical_assertions, organization_options


def sample(quote, statement, verdict=0, evidence_ids=()):
    """Use a scope already located by the evaluator."""
    return {
        "id": "u1",
        "source_unit_id": "unit:0:0",
        "source_fragment_id": "fragment",
        "source_quote": quote,
        "source_start": 0,
        "source_end": len(quote),
        "section": "引言",
        "evidence_ids": list(evidence_ids),
        "statements": [statement],
        "verdicts": (
            [] if verdict is None else [{"reason": "No support.", "verdict": verdict}]
        ),
    }


@pytest.mark.parametrize(
    "profile", [None, "separate-review-evaluation", "llm-stage-repair-stream"]
)
def test_latest_configuration_keeps_embedded_science_unverified_by_host(
    profile, monkeypatch
):
    """The legacy rule could accept a scientific statement with no evidence."""
    scope = "本综述按“视觉表示”解释ViT忽略空间信息"
    evidence_free = sample(scope + "。", scope)
    before, _ = canonical_assertions([evidence_free], host_organization=True)
    assert (before[0]["kind"], before[0]["status"]) == ("report_structure", "supported")

    monkeypatch.delenv("LITAGENT_CONFIG", raising=False)
    config = (
        load_config(live_workbench=True)
        if profile is None
        else load_config(f"config/{profile}.yaml")
    )
    descriptor = configured_engine(config)
    assert descriptor["organization_contract"] == BOUND_ORGANIZATION
    assert validate_engine(descriptor) == descriptor
    forged = {
        **descriptor,
        "identity": "3c117b941d70981dced95f8e7c5e47b8ff73da8ef2713decbb25a63cd3946a9b",
    }
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(forged, archived=True)
    options = organization_options(
        {"semantic_engine": descriptor}, "## 引言\n", {"report_language": "zh-CN"}
    )
    after, _ = canonical_assertions([evidence_free], **options)
    assert (after[0]["kind"], after[0]["status"]) == ("scientific", "unsupported")
    assert after[0]["explanation"] == "No support."


@pytest.mark.parametrize(
    ("quote", "statement", "verdict", "cites", "kind", "status"),
    [
        (
            "本文包含“引言”一节。",
            "本文包含“引言”一节",
            0,
            [],
            "report_structure",
            "supported",
        ),
        (
            "本文包含“引言”一节。",
            "本文包含“引言”一节",
            None,
            [],
            "report_structure",
            "unverifiable",
        ),
        (
            "本文包含“结论”一节。",
            "本文包含“结论”一节",
            0,
            [],
            "scientific",
            "unsupported",
        ),
        (
            "本文包含“引言”一节，ViT忽略空间信息。",
            "本文包含“引言”一节",
            0,
            [],
            "scientific",
            "unsupported",
        ),
        (
            "本文包含“引言”一节[E:e1]。",
            "本文包含“引言”一节",
            0,
            ["e1"],
            "scientific",
            "unsupported",
        ),
        (
            'The report includes the section "Introduction".',
            'The report includes the section "Introduction"',
            0,
            [],
            "report_structure",
            "supported",
        ),
    ],
)
def test_only_exact_heading_checks_in_present_template_pass(
    quote, statement, verdict, cites, kind, status
):
    """Never treat arbitrary model prose, omitted headings or incomplete NLI as passed."""
    report_language = "en" if statement.startswith("The ") else "zh-CN"
    options = organization_options(
        {"semantic_engine": {"organization_contract": BOUND_ORGANIZATION}},
        "## Introduction\n## 引言\n\n```markdown\n## 结论\n```\n",
        {"report_language": report_language},
    )
    rows, keys = canonical_assertions(
        [sample(quote, statement, verdict, cites)], **options
    )
    assert keys == {"u1": ["claim_0"]}
    assert (rows[0]["kind"], rows[0]["status"]) == (kind, status)


@pytest.mark.parametrize(
    "previous",
    [
        ARCHIVED_INLINE_PRE_ORGANIZATION_ENGINE,
        ARCHIVED_INLINE_BOUND_ORGANIZATION_ENGINE,
    ],
)
def test_historical_organization_engine_is_archive_only(previous):
    """Keep the exact old descriptor readable without importing it into new runs."""
    assert validate_engine(previous, archived=True) == previous
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(previous)
    mixed = dict(previous)
    if "organization_contract" in mixed:
        del mixed["organization_contract"]
    else:
        mixed["organization_contract"] = BOUND_ORGANIZATION
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(mixed, archived=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inline", "repair"),
    [(True, False), (False, False), (False, True)],
    ids=["stage-repair", "workbench", "workbench-grounded-replacement"],
)
async def test_uncited_science_disguised_as_organization_reaches_repair(
    case, tmp_path, monkeypatch, inline, repair
):
    """Reject the false exemption in the full composer-to-global request chain."""
    from test_inline_citations import InlineGateway
    from test_separate_review_pipeline import SeparateGateway, execute

    disguised = "本综述按“视觉表示”解释ViT忽略空间信息"
    heading = 'The report includes the section "Introduction"'

    class Gateway(InlineGateway if inline else SeparateGateway):
        def answer(self, payload, native_schema):
            value = super().answer(payload, native_schema)
            if payload.get("requested_body_section") == "introduction":
                if inline:
                    value["units"][0]["text"] += f" {heading}. {disguised}。"
                else:
                    value["units"].extend(
                        {"text": text, "fact_card_refs": [], "paragraph_start": False}
                        for text in (f"{heading}.", f"{disguised}。")
                    )
            return value

    monkeypatch.delenv("LITAGENT_CONFIG", raising=False)
    config = load_config(
        "config/llm-stage-repair.yaml" if inline else None, live_workbench=not inline
    )
    gateway = Gateway(case, reviews=0, keep=not repair and "all")
    if repair:
        import json
        from dataclasses import replace

        from test_native_ragas_evaluation import SemanticTransport

        from litagent.eval.ragas_global_wire import unpack_global_input

        class SupportedCoverage(SemanticTransport):
            async def chat(self, messages, **kwargs):
                response = await super().chat(messages, **kwargs)
                payload = json.loads(messages[1]["content"])
                if "source_units" not in payload:
                    return response
                if "owners" in payload:
                    payload = unpack_global_input(payload)
                valid = [
                    row["claim_id"]
                    for row in payload["assertions"]
                    if row["kind"] == "scientific" and row["status"] == "supported"
                ]
                value = json.loads(response.content)
                for row in value["coverage"]:
                    row.update(claim_ids=valid[:1], answered=bool(valid))
                return replace(response, content=json.dumps(value))

        gateway.semantic = SupportedCoverage()
    report = await execute(
        case,
        tmp_path,
        monkeypatch,
        gateway,
        config=config,
    )
    initial = report["metadata"].get("evaluation_history", {}).get("initial_evaluation")
    details = (initial or report["evaluation"])["faithfulness"]["details"]
    rows = {item["claim_text"].rstrip(".。"): item for item in details["assertions"]}
    assert rows[heading]["kind"] == "report_structure"
    assert rows[disguised]["kind"] == "scientific"
    assert rows[disguised]["reason_code"] == "uncited_claim"
    assert rows[disguised]["evidence_ids"] == []
    assert gateway.global_calls and gateway.repair_count
    if inline or repair:
        assert report["delivery"]["publishable"] is repair
    if repair:
        assert gateway.global_calls >= 2
        assert disguised not in report["survey"]
        assert heading in report["survey"]
        assert report["evaluation"]["faithfulness"]["score"] == 1.0
