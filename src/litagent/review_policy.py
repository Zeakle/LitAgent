"""Version the two editorial loops without reinterpreting historical ledgers."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

CONTRACT = "separate-review-evaluation-v1"
PROFILE = "separate-review-evaluation-v1"
LIMITS_CONTRACT = "configured-run-limits-v6"
LEDGER_CONTRACT = "independent-targeted-repair-v2"
ALLOWANCES = {
    "review_revision": 3,
    "evaluation_revision": 3,
    "capacity": 3,
    "inventory": 3,
}
_partition = ContextVar("litagent_review_partition", default=None)
PARTITION_STAGES = {
    "review_revision": ("repair", "summary"),
    "evaluation_revision": ("repair", "summary", "inventory", "assessment", "global"),
    "capacity": ("repair", "summary"),
    "inventory": ("inventory", "assessment", "global"),
}


def enabled(envelope):
    """Select the new behavior only from a frozen, explicit resource revision."""
    limits = (envelope or {}).get("run_limits")
    return isinstance(limits, dict) and limits.get("contract") == LIMITS_CONTRACT


def parent_stage(stage):
    """Resolve a protected substage to its unchanged wire-size envelope."""
    return stage.split("/", 1)[-1]


def stage_name(envelope, stage, category=None):
    """Route an authored request to its independently reserved budget."""
    if not enabled(envelope) or "/" in stage:
        return stage
    category = category or _partition.get()
    if stage in PARTITION_STAGES.get(category, set()):
        return f"{category}/{stage}"
    return stage


@contextmanager
def partition(category):
    """Keep one transaction's ownership across async child calls."""
    if category not in ALLOWANCES:
        raise ValueError("review_partition_invalid")
    token = _partition.set(category)
    try:
        yield
    finally:
        _partition.reset(token)


def stage_counts(batches):
    """Count every reserved request without charging review edits a full eval."""
    counts = {
        "summary": 1,
        "reviewer": 4,
        "inventory": batches,
        "assessment": batches,
        "global": 1,
    }
    for category, stages in PARTITION_STAGES.items():
        for stage in stages:
            counts[f"{category}/{stage}"] = ALLOWANCES[category] * (
                batches if stage in {"inventory", "assessment"} else 1
            )
    return counts


def format_stages(counts):
    """Reserve Lite, Pro and regeneration for each eligible authored response."""
    return {
        "format_body": ("body", 15),
        **{
            "format_" + stage: (stage, 3 * count)
            for stage, count in counts.items()
            if parent_stage(stage) in {"repair", "summary"}
        },
    }


def response_contract(schema, targets, references):
    """Add explicit dispositions without requiring unchanged text to be echoed."""
    import copy

    from litagent.writing_v2 import _array, _closed

    schema = copy.deepcopy(schema)
    directory = {f"t{i}": tid for i, tid in enumerate(targets, 1)}
    item = _closed(
        {
            "action": {"type": "string", "enum": ["replace", "keep", "cannot_resolve"]},
            "reason": {"type": "string", "minLength": 1, "maxLength": 2000},
            "reference_ids": _array(
                {"type": "string", "enum": list(references)},
                minimum=0,
                maximum=max(1, len(references)),
            ),
        }
    )
    schema["properties"]["target_responses"] = _closed(
        {key: copy.deepcopy(item) for key in directory}
    )
    schema["required"].append("target_responses")
    schema["x-target-directory"] = directory
    return schema


def validate_responses(result, targets, references):
    """Require a complete response and patches only for explicit replacements."""
    from litagent.writing_v2 import WriterError

    directory = {f"t{i}": tid for i, tid in enumerate(targets, 1)}
    responses = result.get("target_responses")
    if not isinstance(responses, dict) or set(responses) != set(directory):
        raise WriterError("writer_target_response_coverage_invalid")
    replaced = set()
    for key, value in responses.items():
        if (
            not isinstance(value, dict)
            or set(value) != {"action", "reason", "reference_ids"}
            or not isinstance(value["action"], str)
            or value["action"] not in {"replace", "keep", "cannot_resolve"}
            or not isinstance(value["reason"], str)
            or not value["reason"].strip()
            or not isinstance(value["reference_ids"], list)
            or any(not isinstance(ref, str) for ref in value["reference_ids"])
            or len(value["reference_ids"]) != len(set(value["reference_ids"]))
            or not set(value["reference_ids"]) <= set(references)
        ):
            raise WriterError("writer_target_response_invalid")
        if value["action"] == "replace":
            replaced.add(directory[key])
        elif value["action"] == "keep" and not value["reference_ids"]:
            # A scientific justification must identify an available premise.
            raise WriterError("writer_keep_basis_missing", "quality")
    ids = [p["target_id"] for p in result["patches"]]
    if len(ids) != len(set(ids)) or set(ids) != replaced:
        raise WriterError("writer_patch_response_mismatch")
    return {tid: targets[tid] for tid in targets if tid in replaced}


@lru_cache(maxsize=1)
def identity():
    """Bind only the opted-in workflow, leaving old profile selectors unchanged."""
    import hashlib
    import inspect
    from pathlib import Path

    from litagent import review_cycles
    from litagent.quality import digest

    return digest(
        [
            CONTRACT,
            LEDGER_CONTRACT,
            ALLOWANCES,
            PARTITION_STAGES,
            *[
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "writing_v2.py",
                    "writing_flat.py",
                    "writer_source_view.py",
                    "review_cycles.py",
                )
            ],
            *[
                inspect.getsource(f)
                for f in (
                    stage_name,
                    stage_counts,
                    format_stages,
                    response_contract,
                    validate_responses,
                    review_cycles.editorial_cycle,
                    review_cycles.improvement,
                )
            ],
        ]
    )


# These exact opt-in deltas are excluded only from pre-v6 source fingerprints.
# Their complete implementations remain bound by the v6 workflow identity.
_LEGACY_SOURCE_DELTAS = {
    "WriterSession._patch": [
        ("    async def _patch(\n", "    async def patch(\n"),
        (
            '                    self.state,\n                    (\n                        (\n                            "review_revision"\n                            if cause == "review"\n                            else (\n                                "evaluation_revision"\n                                if cause == "evaluation"\n                                else "capacity"\n                            )\n                        )\n                        if review_policy.enabled(self.envelope)\n                        else ("prose" if cause == "evaluation" else "capacity")\n                    ),\n',
            '                    self.state, "prose" if cause == "evaluation" else "capacity"\n',
        ),
        (
            '            (\n                "review"\n                if cause == "review"\n                else "evaluation" if cause == "evaluation" else "native_initial"\n            ),\n',
            '            "evaluation" if cause == "evaluation" else "native_initial",\n',
        ),
        (
            '            if review_policy.enabled(self.envelope) and cause != "capacity":\n                entry["target_responses"] = copy.deepcopy(\n                    result.get("target_responses")\n                )\n                entry["all_authorized_targets"] = copy.deepcopy(targets)\n                targets = review_policy.validate_responses(result, targets, references)\n                if not targets:\n                    entry.update(\n                        status="completed",\n                        candidate_sha256=digest(body),\n                        replacement_lengths={},\n                        outcome="unchanged",\n                    )\n                    self.save()\n                    return copy.deepcopy(body)\n',
            "",
        ),
        (
            '                if review_policy.enabled(self.envelope) and cause != "capacity":\n                    entry["provenance_receipt"]["response_targets"] = copy.deepcopy(\n                        entry["all_authorized_targets"]\n                    )\n',
            "",
        ),
    ],
    "check_guidance_wire_record": [
        (
            '        max_depth = (\n            allowances.get("prose", allowances.get("evaluation_revision")) - 1\n            if allowances\n            else 1\n        )\n',
            '        max_depth = allowances["prose"] - 1 if allowances else 1\n',
        )
    ],
    "repair_prompt_system": [
        (
            '        elif roots == {"outcome", "explanation", "patches", "target_responses"}:\n            if fields not in (\n                {"target_id", "text", "fact_card_refs"},\n                {"target_id", "text", "body_unit_refs"},\n            ):\n                raise ValueError("unsupported response fields")\n            output = (\n                "Respond to EVERY target_responses key with replace, keep or cannot_resolve. "\n                "response_target_keys maps short tN keys to authorized targets. "\n                "Only replace has patches; use its short tN key and the schema\'s reference field. "\n                "Keep requires actual frozen reference_ids and a reason; do not echo unchanged text. "\n                "A response is not a scientific verdict. Never silently omit a target or edit outside it."\n            )\n',
            "",
        )
    ],
    "project_source_view": [
        (
            '    from litagent import review_policy\n\n    if "response_target_keys" in payload and not review_policy.enabled(envelope):\n        _invalid()\n',
            "",
        )
    ],
}


def legacy_source(function):
    """Preserve the prior source identity while binding new branches separately."""
    import inspect

    source = inspect.getsource(function)
    for added, original in _LEGACY_SOURCE_DELTAS.get(function.__qualname__, ()):
        if source.count(added) != 1:
            raise ValueError("review_legacy_source_projection_invalid")
        source = source.replace(added, original, 1)
    return source
