"""Authenticate current repair locations without parsing reviewer explanations."""

import copy

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.writing_v2 import (
    WriterError,
    bound_fragments,
    compile_document,
    current_repair_guidance_context,
    repair_targets,
)


@pytest.fixture
def located(case):
    """Compile actual report bindings and one exact scientific diagnostic."""
    survey, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en", {}
    )
    row = next(r for r in bindings["units"] if r["path"][0] == "methods")
    parent, fragment = next(
        (uid, fragment)
        for uid, group in bound_fragments(survey, bindings).items()
        for fragment in group
        if fragment["source_fragment_id"].endswith(":native:" + row["unit_id"])
    )
    diagnostic = {
        "claim_id": "located-science",
        "source_unit_id": parent,
        "source_fragment_id": fragment["source_fragment_id"],
        "source_start": row["source_start"],
        "source_end": row["source_end"],
        "source_quote": row["source_quote"],
        "evidence_ids": row["evidence_ids"],
        "explanation": "A complete local-only explanation.",
    }
    document = {
        "packet": case.packet,
        "aliases": case.aliases,
        "body": case.body,
        "summaries": case.summaries,
    }
    return survey, bindings, row, diagnostic, document


def test_current_context_authenticates_science_and_mechanical_reviewer(case, located):
    survey, bindings, row, science, document = located
    reviewer = {"unit_id": row["unit_id"], "explanation": "Do not infer a type."}
    diagnostics = [science, reviewer]
    targets = repair_targets(bindings, diagnostics, case.body, case.summaries)
    originals = copy.deepcopy((document, diagnostics, targets))
    context = current_repair_guidance_context(
        survey,
        bindings,
        document,
        targets,
        diagnostics,
        diagnostic_categories=["evaluation", "reviewer"],
    )
    directory = context["host_directory"]
    assert context["diagnostics"] == diagnostics
    assert directory["diagnostics_sha256"] == digest(diagnostics)
    assert directory["source_bindings_sha256"] == digest(bindings)
    assert [i["category"] for i in directory["issues"]] == ["evaluation", "reviewer"]
    assert [i["diagnostic_index"] for i in directory["issues"]] == [0, 1]
    assert all(i["target_ids"] for i in directory["issues"])
    assert (document, diagnostics, targets) == originals


@pytest.mark.parametrize(
    "fault", ["report", "binding_type", "quote", "reference", "unknown_fragment"]
)
def test_current_context_rejects_false_locations_and_json_type_aliases(
    case, located, fault
):
    survey, bindings, _, science, document = copy.deepcopy(located)
    targets = repair_targets(bindings, [science], case.body, case.summaries)
    if fault == "report":
        survey += "unbound"
    elif fault == "binding_type":
        bindings["units"][0]["path"][1] = False
    elif fault == "quote":
        science["source_quote"] += " unbound"
    elif fault == "reference":
        science["evidence_ids"] = ["other-paper"]
    else:
        science["source_fragment_id"] += ":unknown"
    with pytest.raises(WriterError):
        current_repair_guidance_context(
            survey,
            bindings,
            document,
            targets,
            [science],
            diagnostic_categories=["evaluation"],
        )


def test_current_context_never_guesses_location_from_free_explanation(case, located):
    survey, bindings, _, science, document = located
    targets = repair_targets(bindings, [science], case.body, case.summaries)
    diagnostic = {"explanation": science["source_quote"], "claim_id": "not-a-location"}
    with pytest.raises(WriterError, match="writer_guidance_unrepairable"):
        current_repair_guidance_context(
            survey,
            bindings,
            document,
            targets,
            [diagnostic],
            diagnostic_categories=["evaluation"],
        )


def test_reportless_local_context_has_no_fabricated_report_positions(case):
    target = {
        "local": {
            "path": ["methods", 0, 0],
            "original": copy.deepcopy(case.body["methods"][0]["sentences"][0]),
            "explanation": "Local prose remains outside the guidance contract.",
        }
    }
    document = {"packet": case.packet, "body": case.body, "summaries": case.summaries}
    diagnostics = [{"code": "writer_reference_invalid", "target_ids": ["local"]}]
    context = current_repair_guidance_context(
        None, None, document, target, diagnostics, diagnostic_categories=["local"]
    )
    directory = context["host_directory"]
    assert directory["current_report_sha256"] is None
    assert directory["source_report_sha256"] is None
    assert directory["source_bindings_sha256"] is None
    assert directory["fragments"] == {}
    assert directory["issues"][0]["body_paths"] == [["methods", 0, 0]]
    assert directory["current_body_sha256"] == digest(case.body)
