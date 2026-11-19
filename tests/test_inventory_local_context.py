"""Keep extraction context separate from each occurrence's scientific sources."""

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest
from test_native_ragas_adapter import Transport

from litagent.config import EvalConfig
from litagent.eval.ragas_adapter import (
    INVENTORY_CONTEXT_CONTRACT,
    NativeRagasAdapter,
    batch_request,
    planned_batches,
)
from litagent.eval.ragas_native import source_samples
from litagent.inline_citations import EVIDENCE
from litagent.report_render import build_report_source_units


def samples():
    """Use real source segmentation, including repeated pronouns and an uncited span."""
    survey = (
        "HTR-ViT targets handwriting[E:h]。"
        "它采用 SAM[E:h]。它采用 SAM[E:h]。它改善泛化。\n\n"
        "ViTSTR targets scene text[E:v]。"
    )
    context = {
        "query": "Compare text recognition methods.",
        "evidence": {
            "h": {"paper_id": "HTR", "supporting_text": "HTR-ViT uses SAM."},
            "v": {"paper_id": "ViTSTR", "supporting_text": "ViTSTR reads scenes."},
        },
    }
    return source_samples(survey, context, inventory_context=True)[-1]


def test_occurrences_keep_local_sources_and_positional_context():
    values = samples()
    repeated = [s for s in values if s["response"] == "它采用 SAM。"]
    assert len(repeated) == 2
    assert repeated[0]["source_start"] != repeated[1]["source_start"]
    assert (
        repeated[0]["report_context"]["start"] != repeated[1]["report_context"]["start"]
    )
    messages, _, ids = batch_request(
        "inventory", [{"sample": s} for s in values], inventory_context=True
    )
    payload = json.loads(messages[1]["content"])
    assert len(ids) == len(values) and len(payload["report_contexts"]) == 2
    assert payload["inventory_context_contract"] == INVENTORY_CONTEXT_CONTRACT
    assert "verdict" not in messages[0]["content"]
    assert "context_pool" not in payload
    for sample in repeated:
        job = payload["jobs"][sample["id"]]
        text = payload["report_contexts"][job["report_context_ref"]]
        assert text[job["answer_start"] : job["answer_end"]] == job["answer"]
        assert "HTR-ViT" in text and "ViTSTR" not in text
        assert "[E:" not in text
        assert sample["evidence_ids"] == ["h"]
    uncited = next(s for s in values if s["response"] == "它改善泛化。")
    assert uncited["contexts"] == uncited["evidence_ids"] == []


@pytest.fixture
def vit_attribution():
    """Load the two frozen inventory inputs and their editorial reference examples."""
    path = Path(__file__).with_name("fixtures") / "vit-inventory-attribution.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drift",
    [
        pytest.param(
            {
                "claim_274": "material_scope",
                "claim_500": "subject",
                "claim_501": "subject",
            },
            id="observed",
        ),
        pytest.param({}, id="reviewed-control"),
        pytest.param({"claim_274": "material_scope"}, id="claim_274-material-scope"),
        pytest.param({"claim_500": "subject"}, id="claim_500-threshold-subject"),
        pytest.param({"claim_501": "subject"}, id="claim_501-ranking-subject"),
    ],
)
async def test_vit_scope_and_author_choice_are_distinct_inventory_targets(
    vit_attribution, drift
):
    """Replay each known change independently, without sending the labels to RAGAS."""
    scopes = vit_attribution["scopes"]
    samples, statements = [], {}
    for scope in scopes:
        quote = scope["source_quote"]
        samples.append(
            {
                "id": scope["sample_id"],
                "response": quote,
                "question": vit_attribution["question"],
                "report_context": scope["report_context"],
                "contexts": [],
            }
        )
        statements[quote] = [
            (
                row["observed"]
                if row["claim_id"] in drift
                else row.get("reference", row["observed"])
            )
            for row in scope["statements"]
        ]
    transport = Transport(statements, {})
    adapter = NativeRagasAdapter(
        transport,
        EvalConfig(),
        keyed_assessment=True,
        inline_scopes=True,
        inventory_context=True,
    )
    result = await adapter.score(samples)
    [(payload, _)] = transport.calls
    assert set(payload) == {"jobs", "inventory_context_contract", "report_contexts"}
    observed_drift = {}
    for scope, sample in zip(scopes, result["samples"], strict=True):
        key = scope["sample_id"]
        assert sample["id"] == key
        job = payload["jobs"][key]
        context = scope["report_context"]
        assert job == {
            "question": vit_attribution["question"],
            "answer": scope["source_quote"],
            "answer_start": context["start"],
            "answer_end": context["end"],
            "report_context_ref": context["id"],
        }
        background = payload["report_contexts"][context["id"]]
        assert background == context["text"]
        assert background[context["start"] : context["end"]] == scope["source_quote"]
        for row, text in zip(scope["statements"], sample["statements"], strict=True):
            if "reference" in row and text == row["observed"]:
                observed_drift[row["claim_id"]] = row["dimension"]
            else:
                assert text == row.get("reference", row["observed"])
        assert sample["assessment_origin"] == "host_uncited"
        assert [v["verdict"] for v in sample["verdicts"]] == [0] * len(
            scope["statements"]
        )
    assert observed_drift == drift


def test_vit_attribution_fixture_matches_original_run(vit_attribution):
    """Check the portable examples against their original occurrences when available."""
    root = Path("artifacts/acceptance/vit14-failure-attribution")
    archive = root / "real-evaluation-20261119/result.json"
    if not archive.exists():
        pytest.skip("requires the original local ViT evaluation")
    result = json.loads(archive.read_bytes())
    survey = (root / "candidate.md").read_text(encoding="utf-8")
    assert (
        sha256(survey.encode()).hexdigest()
        == result["report_sha256"]
        == vit_attribution["report_sha256"]
    )
    assert (result["run_id"], result["model"]) == (
        vit_attribution["run_id"],
        vit_attribution["model"],
    )
    details = result["evaluation"]["faithfulness"]["details"]
    assert details["coverage_requirements"][0]["text"] == vit_attribution["question"]
    units = build_report_source_units(survey)
    spans = details["writer_bindings"]["citation_scopes"]
    for scope in vit_attribution["scopes"]:
        start, end = scope["report_range"]
        ((index, unit),) = [
            (i, u) for i, u in enumerate(units) if u.start <= start < end <= u.end
        ]
        (sample_index,) = [
            i
            for i, s in enumerate(spans)
            if (s["source_start"], s["source_end"]) == (start, end)
        ]
        assert scope["sample_id"] == f"u{sample_index}"
        assert (
            survey[start:end]
            == spans[sample_index]["source_quote"]
            == scope["source_quote"]
        )
        assert scope["report_context"] == {
            "id": f"p{index}",
            "text": EVIDENCE.sub("", unit.text),
            "start": len(EVIDENCE.sub("", survey[unit.start : start])),
            "end": len(EVIDENCE.sub("", survey[unit.start : end])),
        }
        rows = [
            row
            for row in details["assertions"]
            if row["source_fragment_id"] == scope["source_fragment_id"]
        ]
        for row, example in zip(rows, scope["statements"], strict=True):
            assert (row["claim_id"], row["claim_text"]) == (
                example["claim_id"],
                example["observed"],
            )
            assert (row["source_start"], row["source_end"], row["evidence_ids"]) == (
                start,
                end,
                [],
            )


@pytest.mark.parametrize(
    "fault", ["offset", "bool", "text", "key", "conflict", "duplicate"]
)
def test_invalid_context_fails_before_transport(fault):
    values = samples()[:2]
    if fault == "offset":
        values[0]["report_context"]["start"] += 1
    elif fault == "bool":
        values[0]["report_context"]["start"] = False
    elif fault == "text":
        values[0]["report_context"]["text"] = "changed"
    elif fault == "key":
        values[0]["report_context"]["id"] = "p-1"
    elif fault == "conflict":
        values[1]["report_context"]["text"] += "another background"
    else:
        values[1]["id"] = values[0]["id"]
    with pytest.raises(ValueError, match="ragas_inventory_"):
        planned_batches(
            "inventory", values, EvalConfig(), 20, 6, inventory_context=True
        )


def test_legacy_inventory_and_all_nli_wire_remain_unchanged():
    values = samples()
    jobs = [{"sample": s} for s in values]
    legacy, _, _ = batch_request("inventory", jobs)
    assert "Echo each statement word-for-word" in legacy[0]["content"]
    assert "report_contexts" not in json.loads(legacy[1]["content"])
    for sample in values:
        sample["statements"] = [sample["response"]]
    for keyed in (False, True):
        assert batch_request(
            "assessment", jobs, keyed_assessment=keyed
        ) == batch_request(
            "assessment", jobs, keyed_assessment=keyed, inventory_context=True
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong_subject", [False, True])
async def test_official_consumers_use_context_without_borrowing_evidence(wrong_subject):
    values = samples()
    statements = {s["response"]: [s["response"]] for s in values}
    claim = "ViTSTR uses SAM." if wrong_subject else "HTR-ViT uses SAM."
    statements["它采用 SAM。"] = [claim]
    verdicts = {text: 1 for rows in statements.values() for text in rows}
    verdicts[claim] = 0 if wrong_subject else 1
    transport = Transport(statements, verdicts)
    config = EvalConfig()
    adapter = NativeRagasAdapter(
        transport,
        config,
        keyed_assessment=True,
        inline_scopes=True,
        inventory_context=True,
    )
    plans = {}

    def admission(phase, material):
        plans[phase] = planned_batches(
            phase,
            material,
            config,
            adapter.batch_size,
            6,
            keyed_assessment=True,
            inline_scopes=True,
            inventory_context=True,
        )
        return [ids for ids, *_ in plans[phase]]

    adapter.admission = admission
    result = await adapter.score(values)
    assert result["score"] == (0.4 if wrong_subject else 0.8)
    assert len(transport.calls) == 2
    for phase, (payload, options) in zip(("inventory", "assessment"), transport.calls):
        assert payload == json.loads(plans[phase][0][1][1]["content"])
        assert options["response_format"] == plans[phase][0][2]
    nli = transport.calls[1][0]
    assert "report_contexts" not in nli
    for sample in values:
        if not sample["contexts"]:
            assert sample["id"] not in nli["jobs"]
            assert sample["verdicts"][0]["verdict"] == 0
        else:
            refs = nli["jobs"][sample["id"]]["context_refs"]
            assert [nli["context_pool"][ref] for ref in refs] == sample["contexts"]


def test_vit08_pronoun_context_is_bound_without_rejudging_archive():
    from test_global_status_codes import vit_replay

    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-08/repairs.json.writer-v2.json"
    )
    if not path.is_file():
        pytest.skip("requires the immutable local ViT 08 response")
    original = path.read_bytes()
    candidate = json.loads(original)["candidates"][0]
    context = copy.deepcopy(vit_replay()[5])
    context["writing_bindings"] = candidate["bindings"]
    values = source_samples(candidate["survey"], context, inventory_context=True)[-1]
    target = next(s for s in values if s["source_start"] == 9932)
    assert target["response"].startswith("它只使用标准 Transformer encoder")
    assert "HTR-ViT" in target["report_context"]["text"]
    assert "ViTSTR" not in target["report_context"]["text"]
    assert target["evidence_ids"] == ["arxiv:2409.08573:abstract:claim:d306b9898a3e"]
    assert candidate["committed"] is False
    assert path.read_bytes() == original
