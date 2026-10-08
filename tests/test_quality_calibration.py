"""Verify source-grounded calibration data and honest human-review metrics."""

import hashlib
from copy import deepcopy

import pytest

from scripts.build_quality_calibration import build_dataset, calibration_metrics

DOMAINS = {
    "vit": "Vision Transformer ViT",
    "nerf": "NeRF radiance fields",
    "fewshot": "few-shot image classification",
}


def source_artifact(
    domain, *, count=3, run_id=None, shared_papers=(), assertions=True, storage="nodes"
):
    """Mirror RunArtifact source spans and extraction nodes using synthetic papers."""
    run_id = run_id or f"{domain}-run"
    survey = "# Source review\n\n## Evidence\n\n"
    rows, evidence = [], []
    for index in range(count):
        paper = (
            shared_papers[index % len(shared_papers)]
            if shared_papers
            else f"paper:{run_id}:{index}"
        )
        eid = f"{run_id}:claim:{index}"
        quote = f"Model {domain}-{index} reaches {70 + index}% accuracy on Dataset-{index} [E:{eid}]."
        start = len(survey)
        survey += quote + "\n\n"
        rows.append(
            {
                "claim_id": f"claim_{index}",
                "kind": "scientific",
                "claim_text": quote,
                "source_unit_id": f"unit:{index}:{start}",
                "source_fragment_id": f"unit:{index}:fragment:0",
                "source_quote": quote,
                "source_start": start,
                "source_end": start + len(quote),
                "section": "Evidence",
                "evidence_ids": [eid],
                "status": "supported",
            }
        )
        evidence.append(
            {
                "evidence_id": eid,
                "paper_id": paper,
                "paper_title": f"Original {domain} paper {index}",
                "text": quote.split(" [E:")[0],
                "supporting_text": quote.split(" [E:")[0]
                + " under the stated conditions.",
                "source_locator": f"paper={paper};chunk=abstract;section=abstract;page=;block=",
            }
        )
    extraction = {
        "paper_id": "mixed",
        "claims": [item["text"] for item in evidence],
        "evidence_items": evidence,
    }
    artifact = {
        "version": 1,
        "run_id": run_id,
        "query": f"Review {DOMAINS[domain]}",
        "status": "completed",
        "report": {
            "survey": survey,
            "metadata": {},
            "evaluation": {
                "faithfulness": {"details": {"assertions": rows if assertions else []}}
            },
        },
    }
    if storage == "nodes":
        artifact["nodes"] = {
            "worker:extract": {
                "task_id": "extract",
                "kind": "worker",
                "output": [extraction],
            }
        }
    else:
        artifact["results"] = {"extract": [extraction]}
    return artifact


@pytest.mark.parametrize("storage", ["nodes", "results"])
@pytest.mark.parametrize("use_assertions", [True, False])
def test_calibration_samples_round_trip_to_exact_source_and_support(
    storage, use_assertions
):
    """Offset spans and support must refer to actual immutable artifact contents."""
    artifact = source_artifact("vit", storage=storage, assertions=use_assertions)
    source_path = "fixture/vit-source.json"
    before = deepcopy(artifact)
    dataset = build_dataset([(source_path, artifact)], per_domain=40)
    assert dataset["samples"]
    survey = artifact["report"]["survey"]
    sha = hashlib.sha256(survey.encode("utf-8")).hexdigest()
    evidence = (
        artifact["nodes"]["worker:extract"]["output"]
        if storage == "nodes"
        else artifact["results"]["extract"]
    )[0]["evidence_items"]
    by_id = {item["evidence_id"]: item for item in evidence}
    for sample in dataset["samples"]:
        assert sample["source_artifact"] == source_path
        assert sample["source_report_sha256"] == sha
        assert (
            survey[sample["source_start"] : sample["source_end"]]
            == sample["source_quote"]
        )
        assert sample["source_quote"]
        assert set(sample["evidence_ids"]).issubset(by_id)
        for support in sample["original_supports"]:
            source = by_id[support["evidence_id"]]
            for key in ("paper_id", "text", "supporting_text", "source_locator"):
                assert support[key] == source[key]
    assert artifact == before


def test_all_actual_citations_are_preserved_even_when_assertion_mentions_one():
    """The quote's citation closure cannot be narrowed by model extraction IDs."""
    artifact = source_artifact("vit", count=2)
    survey = artifact["report"]["survey"]
    rows = artifact["report"]["evaluation"]["faithfulness"]["details"]["assertions"]
    first, second = rows
    old = first["source_quote"]
    new = old.replace("].", f"] [E:{second['evidence_ids'][0]}].")
    survey = survey.replace(old, new)
    first.update(source_quote=new, source_end=first["source_start"] + len(new))
    artifact["report"]["survey"] = survey
    # The test intentionally keeps only the first source assertion: its declared
    # evidence_ids omits a real neighboring marker in the same exact source span.
    artifact["report"]["evaluation"]["faithfulness"]["details"]["assertions"] = [first]
    dataset = build_dataset([("fixture/vit-multicite.json", artifact)], per_domain=1)
    sample = dataset["samples"][0]
    expected = {first["evidence_ids"][0], second["evidence_ids"][0]}
    assert set(sample["evidence_ids"]) == expected
    assert {item["evidence_id"] for item in sample["original_supports"]} == expected


def test_same_report_and_connected_papers_never_cross_development_holdout():
    """Cross-domain shared papers connect whole reports into a single split group."""
    artifacts = [
        (
            "fixture/vit-a.json",
            source_artifact("vit", run_id="a", shared_papers=("shared:ab",)),
        ),
        (
            "fixture/nerf-b.json",
            source_artifact(
                "nerf", run_id="b", shared_papers=("shared:ab", "shared:bc")
            ),
        ),
        (
            "fixture/fewshot-c.json",
            source_artifact("fewshot", run_id="c", shared_papers=("shared:bc",)),
        ),
        ("fixture/vit-independent.json", source_artifact("vit", run_id="independent")),
    ]
    dataset = build_dataset(artifacts, per_domain=40)
    report_splits, paper_splits = {}, {}
    for sample in dataset["samples"]:
        assert sample["split"] in {"dev", "holdout"}
        report_splits.setdefault(sample["source_report_sha256"], set()).add(
            sample["split"]
        )
        for support in sample["original_supports"]:
            paper_splits.setdefault(support["paper_id"], set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in report_splits.values())
    assert all(len(splits) == 1 for splits in paper_splits.values())
    connected = {
        sample["split"]
        for sample in dataset["samples"]
        if sample["source_artifact"] in {p for p, _ in artifacts[:3]}
    }
    assert len(connected) == 1


def test_duplicate_artifact_sources_do_not_create_duplicate_sample_identity():
    """Identical report sources have stable unique samples across path copies."""
    artifact = source_artifact("vit")
    once = build_dataset([("fixture/original.json", artifact)], per_domain=40)
    twice = build_dataset(
        [
            ("fixture/original.json", artifact),
            ("fixture/copied.json", deepcopy(artifact)),
        ],
        per_domain=40,
    )
    first_ids = [sample["sample_id"] for sample in once["samples"]]
    second_ids = [sample["sample_id"] for sample in twice["samples"]]
    assert len(first_ids) == len(set(first_ids))
    assert len(second_ids) == len(set(second_ids))
    assert set(first_ids) == set(second_ids)


def test_missing_domain_material_is_reported_without_fabricating_120_rows():
    """A target quota remains a shortfall until real source material is available."""
    dataset = build_dataset(
        [("fixture/one.json", source_artifact("vit", count=1))], per_domain=40
    )
    assert len(dataset["samples"]) < 120
    counts = dataset["summary"]["per_domain_counts"]
    shortfall = dataset["summary"]["shortfall"]
    for domain in DOMAINS:
        assert counts[domain] == sum(
            sample["domain"] == domain for sample in dataset["samples"]
        )
        assert shortfall[domain] == 40 - counts[domain]
    assert counts["nerf"] == counts["fewshot"] == 0
    assert shortfall["nerf"] == shortfall["fewshot"] == 40


def test_three_domains_respect_requested_quota_and_all_generated_labels_are_pending():
    """Gathering data or constructing a counterexample never acts as human review."""
    artifacts = [
        (f"fixture/{domain}.json", source_artifact(domain, count=45))
        for domain in DOMAINS
    ]
    dataset = build_dataset(artifacts, per_domain=40)
    assert dataset["summary"]["per_domain_counts"] == {domain: 40 for domain in DOMAINS}
    assert dataset["summary"]["shortfall"] == {domain: 0 for domain in DOMAINS}
    assert len(dataset["samples"]) == 120
    assert dataset["summary"]["annotation_status"] == "annotation_pending"
    for sample in dataset["samples"]:
        assert sample["annotation_status"] == "annotation_pending"
        assert sample["reviewer"] is None
        assert sample["labels"] == {
            "kind": None,
            "atomic": None,
            "source_faithful": None,
            "support_status": None,
        }
    predictions = {sample["sample_id"]: "supported" for sample in dataset["samples"]}
    metrics = calibration_metrics(dataset, predictions)
    assert metrics["human_reviewed_count"] == 0
    assert metrics["false_accept"] == {"count": 0, "denominator": 0, "rate": None}
    assert metrics["false_reject"] == {"count": 0, "denominator": 0, "rate": None}


def reviewed_sample(
    sample_id,
    gold,
    *,
    status="human_reviewed",
    reviewer="unit-test-reviewer",
    **label_changes,
):
    """Use synthetic unit-test labels, never annotate a real calibration artifact."""
    labels = {
        "kind": "scientific",
        "atomic": True,
        "source_faithful": True,
        "support_status": gold,
    }
    labels.update(label_changes)
    return {
        "sample_id": sample_id,
        "annotation_status": status,
        "reviewer": reviewer,
        "labels": labels,
    }


def test_calibration_metrics_use_only_complete_human_labels_with_correct_denominators():
    """Compute false acceptance among bad gold and false rejection among good gold."""
    dataset = {
        "samples": [
            reviewed_sample("good-pass", "supported"),
            reviewed_sample("good-reject", "supported"),
            reviewed_sample("bad-pass", "unsupported"),
            reviewed_sample("bad-reject", "unsupported"),
            reviewed_sample("unknown-reject", "unverifiable"),
            reviewed_sample("pending", "unsupported", status="annotation_pending"),
            reviewed_sample("incomplete", "unsupported", atomic=None),
            reviewed_sample("no-reviewer", "unsupported", reviewer=""),
            reviewed_sample("invalid-kind", "unsupported", kind="anything-goes"),
        ]
    }
    predictions = {
        "good-pass": "supported",
        "good-reject": "unsupported",
        "bad-pass": "supported",
        "bad-reject": "unsupported",
        "unknown-reject": "unverifiable",
        "pending": "supported",
        "incomplete": "supported",
        "no-reviewer": "supported",
        "invalid-kind": "supported",
    }
    metrics = calibration_metrics(dataset, predictions)
    assert metrics["human_reviewed_count"] == 5
    assert metrics["false_accept"]["count"] == 1
    assert metrics["false_accept"]["denominator"] == 3
    assert metrics["false_accept"]["rate"] == pytest.approx(1 / 3)
    assert metrics["false_reject"]["count"] == 1
    assert metrics["false_reject"]["denominator"] == 2
    assert metrics["false_reject"]["rate"] == pytest.approx(1 / 2)


def test_verified_assertion_span_is_prioritized_before_unrelated_report_units():
    """Calibration first includes the actual scored assertions with valid offsets."""
    artifact = source_artifact("vit", count=3)
    last = artifact["report"]["evaluation"]["faithfulness"]["details"]["assertions"][-1]
    artifact["report"]["evaluation"]["faithfulness"]["details"]["assertions"] = [last]
    dataset = build_dataset(
        [("fixture/selected-assertion.json", artifact)], per_domain=1
    )
    assert len(dataset["samples"]) == 1
    sample = dataset["samples"][0]
    assert sample["source_quote"] == last["source_quote"]
    assert sample["source_start"] == last["source_start"]
    assert sample["source_end"] == last["source_end"]


def test_invalid_assertion_offsets_never_create_fabricated_source_quotes():
    """Bad extracted source coordinates fall back to verified report fragments."""
    artifact = source_artifact("vit", count=2)
    rows = artifact["report"]["evaluation"]["faithfulness"]["details"]["assertions"]
    rows[0].update(
        source_quote="Fabricated material absent from the report",
        source_start=0,
        source_end=10,
    )
    dataset = build_dataset([("fixture/bad-offset.json", artifact)], per_domain=40)
    survey = artifact["report"]["survey"]
    assert dataset["samples"]
    for sample in dataset["samples"]:
        assert (
            survey[sample["source_start"] : sample["source_end"]]
            == sample["source_quote"]
        )
        assert "Fabricated material" not in sample["source_quote"]


def test_missing_reference_material_is_explicit_instead_of_fabricated_support():
    """Preserve actual unknown citation IDs and disclose absent original material."""
    artifact = source_artifact("vit", count=1)
    row = artifact["report"]["evaluation"]["faithfulness"]["details"]["assertions"][0]
    unknown = "missing:claim:999"
    quote = row["source_quote"].replace("].", f"] [E:{unknown}].")
    artifact["report"]["survey"] = artifact["report"]["survey"].replace(
        row["source_quote"], quote
    )
    row.update(source_quote=quote, source_end=row["source_start"] + len(quote))
    dataset = build_dataset([("fixture/missing-support.json", artifact)], per_domain=1)
    sample = dataset["samples"][0]
    assert unknown in sample["evidence_ids"]
    assert sample["missing_evidence_ids"] == [unknown]
    assert unknown not in {
        support["evidence_id"] for support in sample["original_supports"]
    }


def test_metrics_exclude_invalid_or_missing_predictions_instead_of_assigning_success():
    """A reviewed gold record without an actual valid prediction is not scored."""
    dataset = {
        "samples": [
            reviewed_sample("missing", "supported"),
            reviewed_sample("invalid", "unsupported"),
        ]
    }
    metrics = calibration_metrics(dataset, {"invalid": "passed"})
    assert metrics["human_reviewed_count"] == 0
    assert metrics["false_accept"]["denominator"] == 0
    assert metrics["false_reject"]["denominator"] == 0
