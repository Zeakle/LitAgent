"""Verify the inline writer, local Evaluation inputs, and report projection."""

import copy
import json
from pathlib import Path

import httpx
import pytest
from test_separate_review_pipeline import SeparateGateway, execute
from test_writer_v2 import case  # noqa: F401

from litagent import inline_citations
from litagent.config import load_config
from litagent.eval.ragas_native import source_samples
from litagent.report_render import build_reader_report
from litagent.run_limits import configured_run_limits
from litagent.workbench import ReportVersion
from litagent.writing_flat import parse, prepare, restore_reference_input
from litagent.writing_v2 import (
    WriterError,
    body_section_schema,
    bound_fragments,
    compile_document,
)


def inline_config():
    config = load_config("config/separate-review-evaluation.yaml")
    config.safety.planned_run_limits.citation_contract = inline_citations.CONTRACT
    return config


@pytest.mark.parametrize("cited", [False, True])
def test_summary_example_uses_a_matching_cited_unit(cited):
    from jsonschema import Draft202012Validator

    from litagent.writing_v2 import request_messages, schema_for

    units = {
        "u1": "An uncited introduction.",
        "u2": "A finding[C:c3]。" if cited else "An uncited finding.",
    }
    schema = schema_for("summary", units)
    schema["x-citation-contract"] = inline_citations.CONTRACT
    system = request_messages(
        "", {"body_units": units}, schema, wire_contract="flat-writer-v1"
    )[0]["content"]
    sample = system.split("JSON encoding example only: ", 1)[1].splitlines()[0]
    example = json.loads(sample)
    Draft202012Validator(prepare(schema)["wire_schema"]).validate(example)
    for section in ("abstract", "conclusion"):
        unit = example[section][0]
        assert unit["body_unit_refs"] == (["u2"] if cited else ["u1"])
        assert inline_citations.references(unit["text"]) == (["c3"] if cited else [])
    assert ("Conclusion prose。[C:c3]" in system) == cited
    assert ("Conclusion prose[C:c3]" in sample) == cited
    assert ("[C:c3]" in system) == cited
    assert "Use only keys in body_units\nas references" not in system
    assert "body_unit_refs contains unique keys from body_units only" in system
    assert "never derive a card number from a body unit key" in system


def test_legacy_summary_keeps_host_derived_reference_instructions():
    from litagent.writing_v2 import (
        LIVE_SUMMARY_PROMPT,
        live_summary_messages,
        request_messages,
        schema_for,
    )

    payload = {"body_units": {"u1": "A supported result."}}
    schema = schema_for("summary", payload["body_units"])
    actual = request_messages(
        LIVE_SUMMARY_PROMPT, payload, schema, wire_contract="flat-writer-v1"
    )
    assert actual == live_summary_messages(
        LIVE_SUMMARY_PROMPT, payload, prepare(schema)["wire_schema"]
    )
    assert "Do not write citations in text." in actual[0]["content"]
    assert "Use only keys in body_units\nas references" in actual[0]["content"]
    assert "[C:" not in actual[0]["content"]


@pytest.mark.parametrize("stage", ["summary_repair_3", "summary_repair_4"])
def test_vit08_original_summary_remains_rejected(stage):
    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-08/repairs.json.writer-v2.json"
    )
    if not path.is_file():
        pytest.skip("requires the immutable local ViT 08 response")
    original = path.read_bytes()
    raw = json.loads(original)["calls"][stage]["raw"]
    summaries = json.loads(raw)
    texts = [unit["text"] for units in summaries.values() for unit in units]
    assert len(texts) == 6
    assert any("[C:c48]" in text for text in texts)
    for text in texts:
        issues = inline_citations.position_diagnostics(text)
        assert issues
        with pytest.raises(WriterError, match="writer_inline_citation_without_prose"):
            inline_citations.references(text)
    assert path.read_bytes() == original


def draft(case):
    body, summaries = copy.deepcopy((case.body, case.summaries))
    # This fixture isolates compilation from the full resource plan.
    for section, paragraphs in body.items():
        if section == "title":
            continue
        unit = paragraphs[0]["sentences"][0]
        ref = unit["fact_card_refs"][0]
        unit["text"] = unit["text"].removesuffix(".") + f"[C:{ref}]."
    for paragraphs in summaries.values():
        unit = paragraphs[0]["sentences"][0]
        unit["text"] = (
            "The comparison function is learned"
            f"[C:{case.by_eid['b']}]. A separate uncited result remains visible."
        )
    return body, summaries


def test_inline_roundtrip_preserves_raw_source_marks(case):
    schema = body_section_schema("methods", case.aliases)
    schema["x-citation-contract"] = inline_citations.CONTRACT
    text = "准确率达到 87.7% 😀[C:c1]。"
    raw = json.dumps({"units": [{"text": text, "paragraph_start": True}]})
    options = {
        "reference_contract": "ordered-reference-set-v1",
        "transport_revision": "host-lossless-transport-v1",
    }
    assert set(
        prepare(schema)["wire_schema"]["properties"]["units"]["items"]["properties"]
    ) == {"text", "paragraph_start"}
    result = parse(raw, schema, **options)
    record = result["bindings"]["units"][0]["reference_spans"][0]
    assert raw[record["start"] : record["end"]] == "[C:c1]"
    assert (
        restore_reference_input(
            result["canonical"],
            result["bindings"],
            transport_revision=options["transport_revision"],
        )
        == result["received_canonical"]
    )
    recovered = parse(
        raw + "}",
        schema,
        format_patch=[{"start": len(raw), "end": len(raw) + 1, "replacement": ""}],
        **options,
    )
    assert recovered["canonical"] == result["canonical"]
    assert recovered["bindings"]["units"][0]["reference_spans"] == [record]
    with pytest.raises(WriterError):
        parse(raw.replace("c1]", "c99]"), schema, **options)


def test_same_source_can_support_two_occurrences_in_one_unit(case):
    schema = body_section_schema("methods", {"c1": case.aliases["c1"]})
    schema["x-citation-contract"] = inline_citations.CONTRACT
    text = "First result[C:c1]. Second result[C:c1]."
    raw = json.dumps({"units": [{"text": text, "paragraph_start": True}]})
    result = parse(raw, schema, reference_contract="ordered-reference-set-v1")
    record = result["bindings"]["units"][0]
    assert record["received_reference_ids"] == ["c1", "c1"]
    assert record["reference_spans"][0]["start"] < record["reference_spans"][1]["start"]
    assert result["canonical"]["methods"][0]["sentences"][0]["fact_card_refs"] == ["c1"]


@pytest.mark.parametrize(
    "text",
    [
        "甲[C:c1]仍在实验中。",
        "甲（不适用[C:c1]）。",
        "甲。 [C:c1]",
        "甲[C:c_1]。",
    ],
)
def test_invalid_inline_mark_positions_are_rejected(text):
    with pytest.raises(WriterError):
        inline_citations.references(text, {"c1"})


def test_multilingual_scope_keeps_decimals_and_unrelated_prose():
    text = "CvT (Fig. 2) reaches 87.7%, e.g. on ImageNet[C:c1]; CI2P uses 63.35% fewer FLOPs[C:c2]. ViTSTR has separate tradeoffs."
    rendered, scopes = inline_citations.compile_text(text, {"c1": "a", "c2": "b"})
    assert [scope["evidence_ids"] for scope in scopes] == [["a"], ["b"], []]
    assert scopes[0]["source_quote"].startswith("CvT (Fig. 2)")
    assert scopes[2]["source_quote"] == "ViTSTR has separate tradeoffs."
    assert all(
        rendered[s["source_start"] : s["source_end"]] == s["source_quote"]
        for s in scopes
    )


def test_compilation_and_ragas_share_local_support(case):
    from litagent.closeout import PRESET
    from litagent.eval.engine_policy import configured_engine
    from litagent.output_capacity import configured_output_tokens
    from litagent.writing_v2 import (
        RANGE_WRITING_PLAN,
        source_packet,
        writer_budget_plan,
    )

    cfg = inline_config()
    body, summaries = draft(case)
    packet, aliases = source_packet(
        case.cards,
        case.selection,
        case.packet["query"],
        case.packet["requirements"],
        "en",
    )
    plan = writer_budget_plan(
        packet,
        aliases,
        budget_policy="planned-budget-v7",
        methodology={},
        capacity=cfg.capacity,
        run_limits=configured_run_limits(cfg),
        semantic_engine=configured_engine(cfg),
        output_tokens=configured_output_tokens(cfg),
        **PRESET,
    )
    assert plan["status"] == "admitted", plan.get("reason")
    envelope = plan["envelope"]
    text = "EViT reduces computation[C:c1]. Relation networks learn comparison[C:c2]."
    body["methods"][0]["sentences"][0].update(text=text, fact_card_refs=["c1", "c2"])
    report, bindings = compile_document(
        body,
        summaries,
        case.cards,
        aliases,
        "en",
        {},
        envelope=envelope,
        writing_plan=RANGE_WRITING_PLAN,
    )
    _, _, _, samples = source_samples(
        report,
        {
            "writing_bindings": bindings,
            "evidence": case.ledger,
        },
    )
    methods = [row for row in samples if row["section"] == "Representative Methods"]
    assert [row["evidence_ids"] for row in methods] == [["a"], ["b"]]
    assert "Relation networks" not in methods[0]["contexts"][0]
    assert "EViT" not in methods[1]["contexts"][0]
    assert any(not row["contexts"] and "uncited" in row["response"] for row in samples)
    version = ReportVersion(
        run_id="inline", revision=1, report_language="en", survey=report
    )
    for style in ("ieee", "apa7", "numbered"):
        reader = build_reader_report(
            version,
            evidence_ledger=case.ledger,
            paper_catalog={},
            citation_style=style,
            writing_bindings=bindings,
        )
        assert reader.source_sha256 == version.content_sha256
        first, second = [
            link
            for link in reader.citation_links
            if "reduces computation" in link["citation_scope"]["source_quote"]
            or "networks learn comparison" in link["citation_scope"]["source_quote"]
        ]
        assert first["evidence_ids"] == ["a"]
        assert second["evidence_ids"] == ["b"]
    tampered = copy.deepcopy(bindings)
    tampered["citation_scopes"].pop(0)
    with pytest.raises(WriterError):
        bound_fragments(report, tampered)
    summaries["abstract"][0]["sentences"][0]["body_unit_refs"] = ["u1"]
    summaries["abstract"][0]["sentences"][0]["text"] = "An unsupported transfer[C:c1]."
    with pytest.raises(WriterError, match="writer_inline_citation_unknown"):
        compile_document(
            body,
            summaries,
            case.cards,
            aliases,
            "en",
            {},
            envelope=envelope,
            writing_plan=RANGE_WRITING_PLAN,
        )


class InlineGateway(SeparateGateway):
    """Use the new native schema at the existing mock transport boundary."""

    def answer(self, payload, native_schema):
        result = super().answer(payload, native_schema)

        def marked(unit, refs):
            text = unit["text"]
            ending = text[-1] if text and text[-1] in ".?!。！？" else ""
            base = text[:-1] if ending else text
            unit["text"] = base + "".join(f"[C:{ref}]" for ref in refs) + ending

        for key in ("units", "patches", "abstract", "conclusion"):
            for unit in result.get(key, []):
                refs = unit.pop("fact_card_refs", None)
                if refs is not None:
                    marked(unit, refs)
                elif unit.get("body_unit_refs"):
                    marked(unit, ["c1"])
        return result


@pytest.mark.asyncio
async def test_inline_workbench_writing_format_repair_and_reassessment(
    case, tmp_path, monkeypatch
):
    gateway = InlineGateway(case, reviews=1, format_fault="extra_brace")
    report = await execute(case, tmp_path, monkeypatch, gateway, config=inline_config())
    assert report["delivery"]["publishable"], (
        report["metadata"].get("writer_failure"),
        report["metadata"].get("recovery_decision"),
        report["delivery"],
    )
    assert gateway.editorial_edits == gateway.repair_count == gateway.format_calls == 1
    assert gateway.global_calls == 2
    assert (
        report["metadata"]["writing_bindings"]["citation_contract"] == "flat-writer-v2"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "position",
        "without_prose",
        "mixed",
        "unknown_usage",
        "unknown_card",
        "wrong_dependency",
        "partition",
        "request",
        "binding",
    ],
)
async def test_inline_review_summary_rejection_requires_a_settled_bound_candidate(
    case, tmp_path, monkeypatch, fault
):
    from litagent import review_cycles, writing_v2
    from litagent.safety.planned_budget import planned_backend

    class SummaryGateway(InlineGateway):
        refreshes = 0

        def answer(self, payload, native_schema):
            result = super().answer(payload, native_schema)
            if (
                fault in {"wrong_dependency", "mixed"}
                and payload.get("requested_body_section") == "introduction"
            ):
                result["units"].append(
                    {
                        "text": "The following sections explain the scope.",
                        "paragraph_start": True,
                    }
                )
            return result

        async def __call__(self, request):
            reply = await super().__call__(request)
            if not request.url.path.endswith("/responses"):
                return reply
            wire = json.loads(request.content)
            if (
                self.editorial_edits == 1
                and self.global_calls == 0
                and "abstract"
                in wire.get("text", {})
                .get("format", {})
                .get("schema", {})
                .get("properties", {})
            ):
                self.refreshes += 1
                response = reply.json()
                output = response["output"][-1]["content"][0]
                summary = json.loads(output["text"])
                unit = summary["abstract"][0]
                unit["text"] = (
                    "Relation networks learn a comparison function[C:c99]."
                    if fault == "unknown_card"
                    else "Relation networks learn a comparison function. [C:c1] The study compared methods."
                )
                if fault in {"wrong_dependency", "mixed"}:
                    payload = json.loads(wire["input"][-1]["content"])
                    unit["body_unit_refs"] = [
                        uid
                        for uid, item in payload["body_units"].items()
                        if (item if isinstance(item, str) else item["text"])
                        == "The following sections explain the scope."
                    ]
                    unit["text"] = (
                        "Relation networks learn a comparison function[C:c1]."
                    )
                if fault == "mixed":
                    unit["text"] = (
                        "Relation networks learn a comparison function[C:c1][C:c99]."
                    )
                if fault == "without_prose":
                    unit["text"] = (
                        "Relation networks learn a comparison function。[C:c1]"
                    )
                if fault == "unknown_usage":
                    response.pop("usage")
                output["text"] = json.dumps(summary)
                return httpx.Response(reply.status_code, json=response)
            return reply

    eligibility = review_cycles.settled_rejected_summary
    compile_document = writing_v2.compile_document
    awaiting_retained_compile = False

    def check_eligibility(session, error, body, summaries, **kwargs):
        nonlocal awaiting_retained_compile
        name = "summary_repair_1"
        budget = planned_backend(session.llm._cost_budget)
        settlement = budget.entries[f"{session.identity}:{name}"]
        ledger_field, replacement = {
            "partition": ("stage", "evaluation_revision/summary"),
            "request": ("wire_sha256", "0" * 64),
        }.get(fault, (None, None))
        if ledger_field:
            original = settlement[ledger_field]
            settlement[ledger_field] = replacement
            try:
                assert not eligibility(session, error, body, summaries, **kwargs)
                return False
            finally:
                settlement[ledger_field] = original
        accepted = eligibility(session, error, body, summaries, **kwargs)
        if fault == "binding":
            assert accepted
            awaiting_retained_compile = True
        return accepted

    def compile_with_altered_binding(*args, **kwargs):
        nonlocal awaiting_retained_compile
        draft, bindings = compile_document(*args, **kwargs)
        if awaiting_retained_compile:
            awaiting_retained_compile = False
            bindings["citation_scopes"][0]["source_start"] += 1
        return draft, bindings

    if fault in {"partition", "request", "binding"}:
        monkeypatch.setattr(
            review_cycles, "settled_rejected_summary", check_eligibility
        )
        if fault == "binding":
            monkeypatch.setattr(
                writing_v2, "compile_document", compile_with_altered_binding
            )
    gateway = SummaryGateway(case)
    if fault == "unknown_card":
        reject = review_cycles.reject_summary_candidate

        def checked_rejection(session, error, body, summaries, retained, **kwargs):
            before = copy.deepcopy(session.state)
            owner = planned_backend(session.llm._cost_budget)
            charged = copy.deepcopy(owner.entries)
            for field in ("draft", "evidence_snapshot_sha256", "writing_bindings"):
                altered = copy.deepcopy(retained)
                if field == "writing_bindings":
                    altered[field]["citation_scopes"][0]["source_start"] += 1
                else:
                    altered[field] += "tampered"
                assert (
                    reject(session, error, body, summaries, altered, **kwargs) is None
                )
            answer = reject(session, error, body, summaries, retained, **kwargs)
            assert answer is not None
            assert reject(session, error, body, summaries, retained, **kwargs) == answer
            assert session.state == before and owner.entries == charged
            saved = session.journal["summary_rejections"][answer["call"]]
            saved["retained_report_sha256"] = "0" * 64
            assert reject(session, error, body, summaries, retained, **kwargs) is None
            session.journal["summary_rejections"][answer["call"]] = copy.deepcopy(
                answer
            )
            session.save()
            return answer

        monkeypatch.setattr(
            review_cycles, "reject_summary_candidate", checked_rejection
        )
    report = await execute(case, tmp_path, monkeypatch, gateway, config=inline_config())
    assert gateway.refreshes == 1 and gateway.format_calls == 0
    if fault not in {
        "position",
        "wrong_dependency",
        "unknown_card",
        "without_prose",
        "mixed",
    }:
        assert gateway.global_calls == 0
        assert not report["delivery"]["publishable"]
        return
    metadata = report["metadata"]
    [row] = metadata["editorial_history"]
    assert row["status"] == "candidate_rejected"
    assert row["reason_code"] == (
        "writer_inline_citation_unknown"
        if fault in {"wrong_dependency", "unknown_card", "mixed"}
        else (
            "writer_inline_citation_without_prose"
            if fault == "without_prose"
            else "writer_inline_citation_position_invalid"
        )
    )
    assert (
        row["source_report_sha256"]
        == metadata["evaluation_history"]["initial_evaluation"]["faithfulness"][
            "details"
        ]["source_report_sha256"]
    )
    assert "revision 1" not in report["survey"]
    assert gateway.global_calls >= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["retry", "exhausted", "unknown_usage"])
@pytest.mark.parametrize(
    "fault", ["without_prose", "position", "unknown_card", "wrong_dependency", "mixed"]
)
async def test_inline_evaluation_repair_retries_only_settled_summary_errors(
    case, tmp_path, monkeypatch, mode, fault
):
    class Gateway(InlineGateway):
        def __init__(self):
            super().__init__(case, reviews=0)
            self.repair_inputs = []

        def answer(self, payload, schema):
            value = super().answer(payload, schema)
            if payload.get("requested_body_section") == "introduction":
                value["units"].append(
                    {
                        "text": "The following sections explain the scope.",
                        "paragraph_start": True,
                    }
                )
            return value

        async def __call__(self, request):
            reply = await super().__call__(request)
            if not request.url.path.endswith("/responses") or not self.global_calls:
                return reply
            wire = json.loads(request.content)
            schema = wire["text"]["format"]["schema"]["properties"]
            if "target_responses" in schema:
                self.repair_inputs.append(json.loads(wire["input"][-1]["content"]))
            if "abstract" not in schema or (
                self.repair_count > 1 and mode != "exhausted"
            ):
                return reply
            response = reply.json()
            output = response["output"][-1]["content"][0]
            summary = json.loads(output["text"])
            unit = summary["abstract"][0]
            unit["text"] = {
                "without_prose": "Relation networks learn a comparison function。[C:c1]",
                "position": "Relation networks learn a comparison function[C:c1] extra.",
                "unknown_card": "Relation networks learn a comparison function[C:c99].",
                "wrong_dependency": "Relation networks learn a comparison function[C:c2].",
                "mixed": "Relation networks learn a comparison function[C:c2][C:c99].",
            }[fault]
            if fault in {"wrong_dependency", "mixed"}:
                payload = json.loads(wire["input"][-1]["content"])
                unit["body_unit_refs"] = ["u1"]
                assert payload["body_units"]["u1"]["fact_card_refs"] == ["c1"]
            output["text"] = json.dumps(summary)
            if mode == "unknown_usage":
                response.pop("usage")
            return httpx.Response(200, json=response)

    gateway = Gateway()
    report = await execute(case, tmp_path, monkeypatch, gateway, config=inline_config())
    metadata = report["metadata"]
    attempts = metadata["evaluation_history"]["attempts"]
    if mode == "unknown_usage":
        assert len(attempts) == gateway.repair_count == gateway.global_calls == 1
        assert not report["delivery"]["publishable"]
        return
    limit = 3 if mode == "exhausted" else 2
    assert gateway.repair_count == limit
    assert sum(row["attempted"] for row in attempts) == limit
    if mode == "exhausted":
        assert attempts[-1]["reason_code"] == "ragas_recovery_prose_limit_reached"
        assert not attempts[-1]["attempted"]
    first = attempts[0]
    assert first["reason_code"] == (
        "writer_inline_citation_unknown"
        if fault in {"unknown_card", "wrong_dependency", "mixed"}
        else (
            "writer_inline_citation_without_prose"
            if fault == "without_prose"
            else "writer_inline_citation_position_invalid"
        )
    )
    assert first["summary_rejection"]["action"] == "retry_evaluation_revision"
    assert first["summary_rejection"]["remaining"] == 2
    assert first["candidate"] is None and first["candidate_evaluation"] is None
    assert first["local_diagnostics"]
    assert len({row["body_sha256"] for row in gateway.repair_inputs}) == 1
    assert (
        gateway.repair_inputs[1]["previous_local_failure"]["code"]
        == first["reason_code"]
    )
    assert gateway.global_calls == (1 if mode == "exhausted" else 2)
    assert report["delivery"]["publishable"] is (mode == "retry")


@pytest.mark.asyncio
async def test_inline_uncited_claim_reaches_global_review_and_repair(
    case, tmp_path, monkeypatch
):
    """Exercise real extraction, local assessment, global review, and repair routing."""
    from litagent.eval.ragas_global_wire import unpack_global_input

    unsupported = "Relation networks require 42 GPUs."

    class MixedGateway(InlineGateway):
        def answer(self, payload, native_schema):
            result = super().answer(payload, native_schema)
            if payload.get("requested_body_section") == "methods":
                result["units"][0]["text"] += " " + unsupported
            return result

    gateway = MixedGateway(case, keep="all")
    report = await execute(case, tmp_path, monkeypatch, gateway, config=inline_config())
    evaluation = report["evaluation"]["faithfulness"]
    details = evaluation["details"]
    (uncited,) = [
        row for row in details["assertions"] if row["claim_text"] == unsupported
    ]
    assert uncited["status"] == "unsupported"
    assert uncited["reason_code"] == "uncited_claim"
    assert uncited["assessment_origin"] == "host_uncited"
    assert uncited["source_quote"] == unsupported
    assert not uncited["evidence_ids"]
    assert details["stages"]["global_check"] == "completed"
    assert 0 < evaluation["score"] < 1
    assert not report["delivery"]["publishable"]
    assert gateway.repair_count == 1
    assert details["stages"]["batches"][-1]["origin"] == "host_uncited"
    assert uncited["claim_id"] in details["stages"]["batches"][-1]["claim_ids"]
    nli = [row["payload"] for row in gateway.sent if "context_pool" in row["payload"]]
    assert nli and all(
        job["context_refs"] for payload in nli for job in payload["jobs"].values()
    )
    global_payload = next(
        row["payload"] for row in gateway.sent if "source_units" in row["payload"]
    )
    if "owners" in global_payload:
        global_payload = unpack_global_input(global_payload)
    assert any(
        row["claim_id"] == uncited["claim_id"] and row["status"] == "unsupported"
        for row in global_payload["assertions"]
    )


def test_vit_replay_uses_paper_local_sources():
    """Replay the cited findings from 235d5bf1dba2's second abstract unit."""
    original = (
        "实验证据显示，ViT 的优势与数据和资源条件绑定。"
        "CvT 在 ImageNet-22k 预训练后，CvT-W24 在 ImageNet-1k 验证集达到 87.7% top-1；"
        "CI2P-ViT 在 Animals-10 从零训练时达到 92.37%，较 ViT-B/16 高 3.3 个百分点，"
        "FLOPs 降低 63.35%，相同硬件训练速度提高 2 倍；"
        "ViTSTR 的 Tiny、Small、Base 则在场景文本识别中呈现节省资源、接近基线和"
        "提高准确率但增加参数/FLOPs 的不同取舍。"
    )
    text = (
        original.replace("top-1；", "top-1[C:c26]；")
        .replace("提高 2 倍；", "提高 2 倍[C:c5]；")
        .replace("不同取舍。", "不同取舍[C:c23]。")
    )
    refs = {
        "c26": "arxiv:2103.15808",
        "c5": "arxiv:2502.10120",
        "c23": "arxiv:2105.08582",
    }
    report, scopes = inline_citations.compile_text(text, refs)
    assert inline_citations.EVIDENCE.sub("", report) == original
    assert [scope["evidence_ids"] for scope in scopes] == [
        [],
        ["arxiv:2103.15808"],
        ["arxiv:2502.10120"],
        ["arxiv:2105.08582"],
    ]
    assert "92.37%" not in scopes[1]["source_quote"]
    assert "87.7%" not in scopes[2]["source_quote"]


def test_quote_end_does_not_lend_the_following_citation():
    _, scopes = inline_citations.compile_text(
        "The report states “A distinct result.” The second result is from CvT[C:c1].",
        {"c1": "cvt"},
    )
    assert [scope["evidence_ids"] for scope in scopes] == [[], ["cvt"]]


def test_global_projection_preserves_inline_citation_positions():
    from litagent.eval.ragas_global_wire import pack_global_input, unpack_global_input

    unit = {
        "source_unit_id": "u",
        "text": "甲[E:arxiv:a]。乙[E:arxiv:b]，丙[E:arxiv:a]。",
    }
    payload = {"source_units": [unit], "assertions": [], "review_methodology": {}}
    wire = pack_global_input(payload, inline_sources=True)
    assert wire["source_units"][0]["text"] == "甲[E:@0]。乙[E:@1]，丙[E:@0]。"
    assert unpack_global_input(wire) == payload
    wire["source_units"][0]["text"] += "[E:@2]"
    with pytest.raises(ValueError, match="ragas_global_inline_citation_invalid"):
        unpack_global_input(wire)


def test_inline_engine_and_workflow_have_separate_frozen_identities():
    from litagent.eval.engine_policy import (
        ARCHIVED_INLINE_VIT_ENGINE,
        configured_engine,
        phase_batches,
        validate_engine,
    )
    from litagent.run_limits import validate_run_limits

    old = load_config("config/separate-review-evaluation.yaml")
    current = inline_config()
    old_engine, current_engine = configured_engine(old), configured_engine(current)
    old_limits = configured_run_limits(old)
    current_limits = configured_run_limits(current)
    assert "citation_contract" not in old_engine
    assert "citation_contract" not in old_limits
    assert old_engine["identity"] != current_engine["identity"]
    assert validate_engine(ARCHIVED_INLINE_VIT_ENGINE, archived=True) == (
        ARCHIVED_INLINE_VIT_ENGINE
    )
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(ARCHIVED_INLINE_VIT_ENGINE)
    assert old_limits["workflow_identity"] != current_limits["workflow_identity"]
    with pytest.raises(ValueError, match="review_workflow_identity_invalid"):
        validate_run_limits(
            {**current_limits, "workflow_identity": old_limits["workflow_identity"]}
        )
    with pytest.raises(ValueError, match="ragas_citation_contract_mismatch"):
        phase_batches({"semantic_engine": old_engine, "run_limits": current_limits})


def test_summary_rejection_projection_preserves_reasons_without_private_receipts():
    from litagent.review_cycles import (
        SUMMARY_REJECTION_CONTRACT,
        summary_rejection_projection,
    )

    original = {
        "contract": SUMMARY_REJECTION_CONTRACT,
        "category": "review_revision",
        "reason_code": "writer_inline_citation_unknown",
        "retained_report_sha256": "a" * 64,
        "call": "summary_repair_1",
        "remaining": 2,
        "actual_tokens": 12176,
        "provider_receipt": {"private": "never exported"},
        "diagnostics": [
            {
                "path": ["conclusion", 0, 1],
                "offset": 10,
                "reference": "c29",
                "in_frozen_pool": True,
                "code": "writer_inline_citation_unknown",
            },
            {
                "path": ["conclusion", 0, 1],
                "offset": 20,
                "reference": "c48",
                "in_frozen_pool": False,
                "code": "writer_inline_citation_unknown",
            },
        ],
    }
    projected = summary_rejection_projection(original)
    assert projected["action"] == "evaluate_retained"
    assert [d["reason"] for d in projected["diagnostics"]] == [
        "outside_body_dependencies",
        "unknown_frozen_card",
    ]
    assert "provider_receipt" not in projected
    assert summary_rejection_projection(projected) == projected
    assert summary_rejection_projection({**original, "remaining": True}) is None
    assert (
        summary_rejection_projection({**original, "retained_report_sha256": "bad"})
        is None
    )
    exhausted = summary_rejection_projection(
        {**original, "category": "evaluation_revision", "remaining": 0}
    )
    assert exhausted["action"] == "stop"


def test_rejected_summary_scanner_keeps_multiple_position_errors():
    text = "First claim[C:c1] extra. Second claim[C:c2] extra."
    diagnostics = inline_citations.position_diagnostics(text)
    assert [d["offset"] for d in diagnostics] == [
        text.index("[C:c1]"),
        text.index("[C:c2]"),
    ]
    with pytest.raises(WriterError, match="writer_inline_citation_position_invalid"):
        inline_citations.references(text)
    assert inline_citations.position_diagnostics("第一句[C:c1]。第二句[C:c2]。") == []
    assert inline_citations.position_diagnostics("第一句。[C:c1]。第二句。[C:c2]") == [
        {
            "code": "writer_inline_citation_without_prose",
            "offset": 4,
            "reference": "c1",
        },
        {
            "code": "writer_inline_citation_without_prose",
            "offset": 15,
            "reference": "c2",
        },
    ]
