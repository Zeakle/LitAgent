"""Preserve scientific gates while sharing focused, source-bound writing duties."""

import copy
import hashlib
import json

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent import body_quality, writing_v2
from litagent.body_quality import WRITER_CONTENT_CONTRACT, WRITER_CONTENT_INSTRUCTION
from litagent.eval.batched import global_prompt
from litagent.workbench import RunOptions
from litagent.writing_v2 import (
    BODY,
    INSTRUCTION,
    WriterError,
    WriterSession,
    body_units,
    compile_document,
    request_messages,
    schema_for,
)


@pytest.mark.parametrize("stage", ["body", "summary", "repair"])
def test_shared_content_is_in_the_actual_request_without_mutating_sources(case, stage):
    """No extra source permission or free scope-statement field is introduced."""
    packet = copy.deepcopy(case.packet)
    refs = body_units(case.body) if stage == "summary" else case.aliases
    schema = schema_for(stage, refs, ["target"])
    before = copy.deepcopy(schema)
    messages = request_messages(INSTRUCTION, packet, schema)
    assert messages[0]["content"].count(WRITER_CONTENT_CONTRACT) == 1
    assert WRITER_CONTENT_INSTRUCTION in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == case.packet == packet
    assert schema == before
    assert "scope_notice_id" not in json.dumps(schema)


def test_global_content_rules_require_explicit_versioned_host_bindings():
    """Old evaluation inputs do not silently acquire a new global quality rubric."""
    old = {"body_analysis_contract": "body-analysis-v1"}
    assert WRITER_CONTENT_INSTRUCTION not in global_prompt(old)
    new = {**old, "writer_content_contract": WRITER_CONTENT_CONTRACT}
    assert WRITER_CONTENT_INSTRUCTION in global_prompt(new)
    assert WRITER_CONTENT_INSTRUCTION in global_prompt(
        {"writer_content_contract": WRITER_CONTENT_CONTRACT}
    )
    assert WRITER_CONTENT_INSTRUCTION not in global_prompt(
        {**old, "writer_content_contract": "unknown"}
    )


def test_focused_summary_selection_cannot_remove_or_rewrite_scientific_body(case):
    """Changing summary focus retains every original body requirement and citation."""
    original = copy.deepcopy(case.body)
    long = copy.deepcopy(case.summaries)
    short = copy.deepcopy(case.summaries)
    for key in long:
        unit = long[key][0]["sentences"][0]
        long[key][0]["sentences"] = [
            {
                **copy.deepcopy(unit),
                "text": f"Finding {i}: Relation networks learn a comparison function.",
            }
            for i in range(6)
        ]
    long_report, long_bindings = compile_document(
        case.body, long, case.cards, case.aliases, "en"
    )
    short_report, short_bindings = compile_document(
        case.body, short, case.cards, case.aliases, "en"
    )
    assert case.body == original

    def scientific_body(bindings):
        return [
            (r["path"], r["source_quote"], r["evidence_ids"])
            for r in bindings["units"]
            if r["path"][0] in BODY
        ]

    assert scientific_body(long_bindings) == scientific_body(short_bindings)
    assert long_bindings["report_sha256"] != short_bindings["report_sha256"]
    assert short_bindings["content_contract"] == WRITER_CONTENT_CONTRACT
    # Local capacity is still the existing six units, not the soft focus target.
    assert len(long_bindings["units"]) == len(body_units(case.body)) + 12
    assert (
        long_bindings["report_sha256"]
        == hashlib.sha256(long_report.encode()).hexdigest()
    )
    assert (
        short_bindings["report_sha256"]
        == hashlib.sha256(short_report.encode()).hexdigest()
    )


def test_content_changes_only_v2_identity_and_old_checkpoint_cannot_resume(
    case, tmp_path, monkeypatch
):
    """Content instructions are part of execution identity, never citation preference."""
    legacy = RunOptions()
    options = {
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "review_structure": "academic_v2",
    }
    staged = RunOptions(generation_strategy="body_summary_v1", **options)
    current = RunOptions(generation_strategy="body_summary_v2", **options)
    before = [x.fingerprint() for x in (legacy, staged, current)]
    session = WriterSession(None, tmp_path / "r.json", "r", case.packet, case.aliases)
    session.save()
    monkeypatch.setattr(body_quality, "WRITER_CONTENT_CONTRACT", "next-content")
    monkeypatch.setattr(writing_v2, "WRITER_CONTENT_CONTRACT", "next-content")
    assert [x.fingerprint() for x in (legacy, staged)] == before[:2]
    assert current.fingerprint() != before[2]
    with pytest.raises(WriterError, match="checkpoint_identity_conflict"):
        WriterSession(None, tmp_path / "r.json", "r", case.packet, case.aliases)
