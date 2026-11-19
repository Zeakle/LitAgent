"""Version the two editorial loops without reinterpreting historical ledgers."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

CONTRACT = "separate-review-evaluation-v1"
PROFILE = "separate-review-evaluation-v1"
LIMITS_CONTRACT = "configured-run-limits-v6"
LEDGER_CONTRACT = "independent-targeted-repair-v2"
LEGACY_WORKFLOW_IDENTITY = (
    "a7abfe34f9eaab9f3ad8e11d25e6dc72c933c844bb5a0f122ef163951ea62e22"
)
ALLOWANCES = {
    "review_revision": 3,
    "evaluation_revision": 3,
    "capacity": 3,
    "inventory": 3,
}
_partition = ContextVar("litagent_review_partition", default=None)
_capacity_parent = ContextVar("litagent_capacity_parent", default=None)
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
    from litagent.eval.native_capacity import selected

    if selected(envelope) and category == "capacity":
        if stage == "inventory":
            return "capacity/inventory"
        if stage in {"assessment", "global"}:
            category = _capacity_parent.get()
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


@contextmanager
def capacity_partition(parent):
    """Reextract privately while retaining the unspent parent NLI/global slots."""
    if parent not in (None, "evaluation_revision", "inventory"):
        raise ValueError("native_capacity_parent_invalid")
    token = _capacity_parent.set(parent)
    try:
        with partition("capacity"):
            yield
    finally:
        _capacity_parent.reset(token)


def capacity_parent():
    """Expose the active parent only while staging a capacity transaction."""
    if _partition.get() != "capacity":
        raise ValueError("native_capacity_partition_missing")
    return _capacity_parent.get()


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


def validate_responses(result, targets, references, *, diagnose=False):
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
    if len(ids) != len(set(ids)) or not set(ids) <= set(targets):
        raise WriterError("writer_patch_response_mismatch")
    if set(ids) != replaced:
        diagnostics = []
        if diagnose:
            for key, tid in directory.items():
                if (tid in ids) != (tid in replaced):
                    diagnostics.append(
                        {
                            "code": "writer_patch_response_mismatch",
                            "target_key": key,
                            "target_id": tid,
                            "action": responses[key]["action"],
                            "patch_index": ids.index(tid) if tid in ids else None,
                            "reason": (
                                "patch_for_non_replace"
                                if tid in ids
                                else "missing_replace_patch"
                            ),
                        }
                    )
        raise WriterError("writer_patch_response_mismatch", diagnostics=diagnostics)
    return {tid: targets[tid] for tid in targets if tid in replaced}


@lru_cache(maxsize=2)
def identity(*, inline=False):
    """Bind only the opted-in workflow, leaving old profile selectors unchanged."""
    if not inline:
        return LEGACY_WORKFLOW_IDENTITY
    return "90ebd7071774543313a4e5eba1614455fad65834bebba16eab2f6bba89fd6300"


# These exact opt-in deltas are excluded only from pre-v6 source fingerprints.
# Their complete implementations remain bound by the v6 workflow identity.
_LEGACY_SOURCE_DELTAS = {
    "summary_request_upper_bound": [
        (
            "    if not selected_summary_body_view(envelope) and not inline_citations.selected(\n"
            "        envelope\n"
            "    ):\n",
            "    if not selected_summary_body_view(envelope):\n",
        ),
    ],
    "_summary_request_bound": [
        (
            "    from litagent.writer_source_view import selected_summary_body_view\n",
            "",
        ),
        (
            "        if inline_citations.selected(envelope) and not selected_summary_body_view(\n"
            "            envelope\n"
            "        ):\n"
            "            # Full premises carry two authored indices alongside each unit.\n"
            "            # Reserve longest encodings even when not all can occur together.\n"
            "            path = [max(BODY, key=len), count - 1, count - 1]\n"
            '            payload["body_units"] = {\n'
            '                uid: {**unit, "path": path} for uid, unit in units.items()\n'
            "            }\n",
            "",
        ),
        (
            "        if inline_citations.selected(envelope):\n"
            "            from litagent.format_repair import summary_retry_messages\n\n"
            "            cited = inline_citations.cited_summary_probe(\n"
            "                system, payload, schema, aliases\n"
            "            )\n"
            '            output = envelope_output_tokens(envelope)["summary"]\n'
            "            measured = max(\n"
            "                strict_request_reservation(summary_retry_messages(wire), output, native)\n"
            "                for wire in (messages, cited)\n"
            "                if wire is not None\n"
            "            )\n",
            "",
        ),
    ],
    "project_structured_repair_guidance": [
        (
            '            if (\n                d["origin"] == "current"\n                and native[0] in current_body\n                and ":scope:" in fid.split(":native:", 1)[0]\n                and native[0] in w.BODY\n            ):\n                unit = current_body[native[0]][native[1]]["sentences"][native[2]]\n                index = int(fid.split(":native:", 1)[0].rsplit(":scope:", 1)[1])\n                _, ranges = inline_citations.compile_text(\n                    unit["text"],\n                    {ref: cards[ref]["evidence_id"] for ref in unit["fact_card_refs"]},\n                )\n                if (\n                    not 0 <= index < len(ranges)\n                    or ranges[index]["evidence_ids"] != fragment["evidence_ids"]\n                    or ranges[index]["source_quote"] != fragment["text"]\n                ):\n                    fail("fragment_body_identity_invalid", identity=True)\n            if (\n                d["origin"] == "current"\n                and native[0] in current_body\n                and ":scope:" not in fid.split(":native:", 1)[0]\n            ):\n',
            '            if d["origin"] == "current" and native[0] in current_body:\n',
        ),
    ],
    "WriterSession._patch": [
        (
            '            # BEGIN native capacity provenance\n            from litagent.eval.native_capacity import (\n                selected as native_capacity_selected,\n            )\n\n            if (\n                native_capacity_selected(self.envelope)\n                and (capacity_diagnostic or {}).get("reason")\n                == "evaluation_native_capacity_refinement_required"\n            ):\n                # Bind the parent before sending. Completed repair entries are\n                # immutable; a later summary must not rewrite their identity.\n                entry["native_capacity"] = {\n                    "budget_partition": review_policy.capacity_parent(),\n                    "source_report_sha256": capacity_diagnostic["source_report_sha256"],\n                    "capacity_identity": capacity_diagnostic["allocation_identity"],\n                }\n            # END native capacity provenance\n',
            "",
        ),
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
        ),
        (
            '    from litagent import stage_repair\n\n    if stage == "summary" and stage_repair.enabled(envelope):\n        # The summary compiler supplies the body-only instruction. Reapplying\n        # source grounding here would change the final request during receipt binding.\n        if GROUNDING_INSTRUCTION in system:\n            _invalid()\n        return system, result\n',
            "",
        ),
    ],
}


def legacy_source(function):
    """Preserve the prior source identity while binding new branches separately."""
    import inspect

    source = inspect.getsource(function)
    from litagent.legacy_stage_sources import restore_source

    source = restore_source(function.__qualname__, source)
    if function.__qualname__ == "WriterSession._patch":
        for added, original in _PATCH_SOURCE_DELTAS:
            if source.count(added) != 1:
                raise ValueError("patch_legacy_source_projection_invalid")
            source = source.replace(added, original, 1)
    for added, original in _LEGACY_SOURCE_DELTAS.get(function.__qualname__, ()):
        if source.count(added) != 1:
            raise ValueError("review_legacy_source_projection_invalid")
        source = source.replace(added, original, 1)
    return source


_PATCH_SOURCE_DELTAS = [
    (
        '        # Bind a rejection before applying any candidate; successful-patch\n        # provenance does not exist for a cross-field response conflict.\n        if inline_citations.selected(self.envelope) and cause in {\n            "review",\n            "evaluation",\n        }:\n            entry["patch_request"] = {\n                "contract": "bound-patch-rejection-v1",\n                "body_sha256": digest(body),\n                "summary_only": summary_body is not None,\n                "targets": copy.deepcopy(targets),\n                "references": list(references),\n                "schema": copy.deepcopy(schema),\n                "messages": copy.deepcopy(messages),\n                "response_format": copy.deepcopy(rf),\n            }\n            entry["patch_request_sha256"] = digest(entry["patch_request"])\n',
        "",
    ),
    (
        "                targets = review_policy.validate_responses(\n                    result,\n                    targets,\n                    references,\n                    diagnose=inline_citations.selected(self.envelope),\n                )\n",
        "                targets = review_policy.validate_responses(result, targets, references)\n",
    ),
]
