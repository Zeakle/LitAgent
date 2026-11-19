"""Compile inline source marks into report text and local citation ranges."""

import hashlib
import json
import re

CONTRACT = "flat-writer-v2"
MARK = re.compile(r"\[C:(c[1-9][0-9]*)\]")
EVIDENCE = re.compile(r"\[E:([^\[\]\s;,]+)\]")
MARK_START = re.compile(r"\[\s*[Cc]\s*:")
BOUNDARY = frozenset("。！？.!?；;")
SEPARATOR = BOUNDARY | frozenset("，,")
ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|et al|vs|cf|Fig|Figs|Eq|Eqs|Sec|Dr|Mr|Ms|Prof|No)|\b[A-Z])\.$",
    re.IGNORECASE,
)
INSTRUCTION = (
    "Use inline source marks such as [C:c3] immediately after the complete "
    "sentence or clause they support, before its punctuation. Copy only supplied "
    "card keys. Put different papers' independent results in distinct clauses, "
    "each with its own marks. Adjacent marks may jointly support one clause. "
    "Use no separate fact_card_refs reply field. Every uncited statement is "
    "still assessed. In summaries, also give body_unit_refs and choose only "
    "the specific marks in those premises. Do not inherit their other sources."
)


def selected(envelope):
    """Read the frozen server selection."""
    return (envelope or {}).get("run_limits", {}).get("citation_contract") == CONTRACT


def identity():
    """Bind new request and citation behavior without changing old source hashes."""
    return "7d262f6c05498461f71700652a202175fce5b0c0bfb74c1a2c1fced29ba3c0cf"


def invalid(code, **diagnostic):
    from litagent.writing_v2 import WriterError

    raise WriterError(code, diagnostics=[diagnostic] if diagnostic else ())


def _segments(text, pattern, *, errors=None, scope_contract=None):
    """Yield ordered spans with citations attached only within a local clause."""
    if scope_contract is not None:
        yield from _local_segments(text, pattern, scope_contract)
        return
    from litagent.eval.scoped import _source_boundary_tokens

    def reject(code, **fields):
        if errors is None:
            invalid(code, **fields)
        errors.append({"code": code, **fields})

    matches = list(pattern.finditer(text))
    marker_at = {m.start(): m for m in matches}
    # The shared scanner already understands Markdown escapes, quotes and
    # brackets. Replace marks with equal-length syntax for its lexical pass.
    probe = text if pattern is EVIDENCE else MARK.sub(lambda m: "[E:" + m[0][3:], text)
    tokens = _source_boundary_tokens(probe, quoted_ends=True)
    punctuation = {pos for pos, token in tokens if token in SEPARATOR}
    ends, seen = set(), set()
    for index, (pos, token) in enumerate(tokens):
        if pos in marker_at and token.startswith("[E:"):
            seen.add(pos)
            after = marker_at[pos].end()
            while after < len(text) and text[after] in " \t":
                after += 1
            if after in marker_at:
                continue
            if after == len(text):
                continue
            if after not in punctuation:
                reject("writer_inline_citation_position_invalid", offset=pos)
            # Treat punctuation just after the marks as part of the same span.
            # The scanner supplies the cut below, including comma-separated clauses.
        elif token in BOUNDARY:
            if token == "." and ABBREVIATION.search(text[: pos + 1]):
                continue
            ends.add(pos + 1)
        elif token in {",", "，"} and index:
            previous, _ = tokens[index - 1]
            if (
                previous in marker_at
                and not text[marker_at[previous].end() : pos].strip()
            ):
                ends.add(pos + 1)
    if seen != marker_at.keys():
        if errors is None:
            invalid("writer_inline_citation_position_invalid")
        for pos in sorted(marker_at.keys() - seen):
            reject("writer_inline_citation_position_invalid", offset=pos)
    start = 0
    for end in sorted(ends | {len(text)}):
        if text[start:end].strip():
            yield start, end, list(pattern.finditer(text, start, end))
        start = end


def _local_segments(text, pattern, contract):
    """End local support at a marker group without borrowing across sentences."""
    from litagent.eval.scoped import _source_boundary_tokens
    from litagent.stage_repair import SCOPE_CONTRACT

    if contract != SCOPE_CONTRACT:
        invalid("writer_inline_scope_contract_invalid")
    matches = list(pattern.finditer(text))
    by_start = {m.start(): m for m in matches}
    probe = text if pattern is EVIDENCE else MARK.sub(lambda m: "[E:" + m[0][3:], text)
    tokens = _source_boundary_tokens(probe, quoted_ends=True)
    visible = {p for p, t in tokens if t.startswith("[E:")}
    if visible != by_start.keys():
        invalid("writer_inline_citation_position_invalid")
    cuts = {len(text)}
    for pos, token in tokens:
        if token in BOUNDARY:
            if token != "." or not ABBREVIATION.search(text[: pos + 1]):
                cuts.add(pos + 1)
    for mark in matches:
        end = mark.end()
        while end < len(text) and text[end] in " \t":
            end += 1
        if end in by_start:
            continue
        # Delimiters belong to the preceding range; prose after them never does.
        while end < len(text) and text[end] in "。！？.!?；;，,、 \t":
            end += 1
        cuts.add(end)
    spans, start = [], 0
    for end in sorted(cuts):
        snippet = text[start:end]
        if not snippet.strip():
            continue
        if (
            spans
            and not pattern.sub("", snippet).strip("。！？.!?；;，,、 \t")
            and not pattern.search(snippet)
        ):
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
        start = end
    for start, end in spans:
        yield start, end, list(pattern.finditer(text, start, end))


def position_diagnostics(text, *, scope_contract=None):
    """Collect syntax-position findings for a rejected candidate only."""
    if MARK_START.search(MARK.sub("", text)) or EVIDENCE.search(text):
        invalid("writer_inline_citation_syntax_invalid")
    errors = []
    for start, end, marks in _segments(
        text, MARK, errors=errors, scope_contract=scope_contract
    ):
        if marks and not MARK.sub("", text[start:end]).strip(
            " \t。！？.!?；;，," + ("、" if scope_contract else "")
        ):
            errors.append(
                {"code": "writer_inline_citation_without_prose", "offset": start}
            )
    by_offset = {m.start(): m[1] for m in MARK.finditer(text)}
    for error in errors:
        offset = error["offset"]
        while offset < len(text) and text[offset] in " \t":
            offset += 1
        if offset in by_offset:
            error["reference"] = by_offset[offset]
    return errors


def references(text, allowed=None, *, scope_contract=None):
    """Validate short marks and return their first occurrence order."""
    if MARK_START.search(MARK.sub("", text)) or EVIDENCE.search(text):
        invalid("writer_inline_citation_syntax_invalid")
    refs = []
    for start, end, marks in _segments(text, MARK, scope_contract=scope_contract):
        if marks and not MARK.sub("", text[start:end]).strip(
            " \t。！？.!?；;，," + ("、" if scope_contract else "")
        ):
            invalid("writer_inline_citation_without_prose", offset=start)
        for mark in marks:
            ref = mark[1]
            if allowed is not None and ref not in allowed:
                invalid(
                    "writer_inline_citation_unknown", reference=ref, offset=mark.start()
                )
            refs.append(ref)
    return list(dict.fromkeys(refs))


def compile_text(text, evidence_by_ref, *, scope_contract=None):
    """Return canonical Markdown and its exact local source spans."""
    references(text, evidence_by_ref, scope_contract=scope_contract)
    chunks, spans, offset = [], [], 0
    for start, end, marks in _segments(text, MARK, scope_contract=scope_contract):
        snippet = text[start:end]
        rendered = MARK.sub(lambda m: f"[E:{evidence_by_ref[m[1]]}]", snippet)
        left = len(snippet) - len(snippet.lstrip())
        right = len(rendered.rstrip())
        spans.append(
            {
                "text_start": start + left,
                "text_end": end - (len(snippet) - len(snippet.rstrip())),
                "source_start": offset + left,
                "source_end": offset + right,
                "source_quote": rendered[left:right],
                "evidence_ids": list(
                    dict.fromkeys(evidence_by_ref[m[1]] for m in marks)
                ),
            }
        )
        chunks.append(rendered)
        offset += len(rendered)
    return "".join(chunks), spans


def raw_occurrences(raw, text, span, path):
    """Map decoded marker offsets to their exact original JSON character offsets."""
    token = raw[span["start"] : span["end"]]
    if json.loads(token) != text:
        invalid("writer_inline_citation_raw_binding_invalid")
    # JSON escapes can represent either one code point or a UTF-16 pair.
    encoded_chars = re.finditer(
        r"\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\.|[^\\]",
        token[1:-1],
        re.DOTALL,
    )
    offsets = [span["start"] + 1]
    offsets.extend(span["start"] + 1 + m.end() for m in encoded_chars)
    if len(offsets) != len(text) + 1:
        invalid("writer_inline_citation_raw_binding_invalid")
    return [
        {
            "wire_path": [*path, "text"],
            "start": offsets[m.start()],
            "end": offsets[m.end()],
        }
        for m in MARK.finditer(text)
    ]


def native_unit_id(fragment_id):
    """Recover the owning editable unit from either report-fragment version."""
    return fragment_id.split(":native:", 1)[-1]


def native_unit_counts(counts):
    """Aggregate several verified ranges that belong to one edit target."""
    result = {}
    for key, count in counts.items():
        uid = native_unit_id(key)
        result[uid] = result.get(uid, 0) + count
    return result


def annotate(schema, envelope):
    """Carry the contract into the sender, parser, and persisted replay schema."""
    if selected(envelope):
        schema["x-citation-contract"] = CONTRACT
        from litagent.stage_repair import scope

        if scope(envelope):
            schema["x-citation-scope-contract"] = scope(envelope)
    return schema


def wire_schema(schema):
    """Remove the host-derived reference field only from authored wire items."""
    import copy

    result = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if (
                "text" in node.get("properties", {})
                and "fact_card_refs" in node["properties"]
            ):
                node["properties"].pop("fact_card_refs")
                node["required"].remove("fact_card_refs")
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(result)
    return result


def scope_instruction(system):
    """Describe the same local range semantics used by the host compiler."""
    return system.replace(
        "immediately after the complete sentence or clause they support, before its punctuation",
        "after the supported words and before any sentence-ending punctuation",
    ).replace(
        "The host rejects marks placed after the clause's punctuation.",
        "A full stop before a citation closes the preceding sentence; the citation "
        "then has no prose in its own scope and cannot support that sentence.",
    ) + (
        "\nEach adjacent citation group ends its own local support span. "
        "A comma, enumeration separator or connective may follow it. "
        "Sentence boundaries and semicolons close a scope before any following marks. "
        "Scope-closing punctuation: "
        + " ".join(sorted(BOUNDARY))
        + ". Decimal points and abbreviations are not sentence boundaries. "
        "Each scientific sentence needs its own actual sources, even within one unit; "
        "a citation at the end of a multi-sentence unit cannot support earlier sentences. "
        "Text after the group needs its own sources; never borrow across a sentence. "
        "candidate_feedback describes a rejected untrusted candidate, not scientific evidence. "
        "Its failed_regions are verbatim excerpts from the indicated diagnostic field, "
        "with character offsets; they are not proposed replacements. "
        "Correct the failed stage while preserving authorized targets and source restrictions."
    )


def repair_system(schema):
    """Retain the repair permissions and use the actual new reply shape."""
    from litagent.writing_flat import (
        _repair_prompt_type_map,
        prepare,
        repair_prompt_system,
    )

    base = repair_prompt_system(
        {k: v for k, v in schema.items() if k != "x-citation-contract"}
    )
    base = base.split("Native output type map (types only, never reply content): ", 1)[
        0
    ]
    base = base.replace(
        "patch.text is plain prose: no LF/headings/HTML/citation tokens. Ref IDs "
        "once/unit; host validates all before ordered-set handling of exact "
        "repeats, preserving original occurrences.",
        "patch.text is prose with local source marks and no LF/headings/HTML.",
    ).replace(
        "use its short tN key and the schema's reference field.",
        "use its short tN key and the fields in the schema.",
    )
    result = (
        base
        + INSTRUCTION
        + "\nDisposition example: if t1=replace, t2=keep, t3=cannot_resolve, "
        "answer all three target_responses but include only t1 in patches. "
        "Never echo kept text in patches. This example supplies no edit authority."
        + "\nFor an l row, complete_prose is the exact local report passage "
        "with its citations in place. Its cited_card_refs resolve the local sources.\n"
        + "Native output type map (types only, never reply content): "
        + _repair_prompt_type_map(prepare(schema)["wire_schema"])
        + ".\n"
    )
    return (
        scope_instruction(result) if schema.get("x-citation-scope-contract") else result
    )


def messages(system, payload, schema, plan):
    """Build stage-specific requests from the same projected schema as dispatch."""
    from litagent.writer_source_view import (
        FINAL_CHAPTER_INSTRUCTION,
        GROUNDING_INSTRUCTION,
    )
    from litagent.writing_v2 import LIVE_SUMMARY_PROMPT

    if plan.get("citation_scope_contract"):
        from litagent.stage_repair import feedback_payload

        payload = feedback_payload(payload)
    original_system = system
    if plan["kind"] == "chapter":
        system = FINAL_CHAPTER_INSTRUCTION.replace(
            "Ordered units have text, fact_card_refs and boolean paragraph_start;",
            "Ordered units have text and boolean paragraph_start;",
        ).replace(
            "Use unique refs inside each unit; never write citation markers in text. The host renders them.\n"
            "[] permits organizational transitions, not unsupported science.",
            "Organizational transitions may be uncited.",
        )
        system += GROUNDING_INSTRUCTION + "\n" + INSTRUCTION
    elif plan["kind"] == "summary":
        system = LIVE_SUMMARY_PROMPT.replace(
            "Use only keys in body_units\nas references and write each reference once per unit.",
            "body_unit_refs contains unique keys from body_units only. "
            "These keys identify content dependencies, not source cards. "
            "In text, copy [C:cN] source marks only from the selected body units; "
            "never derive a card number from a body unit key.",
        ).replace(" Do not write citations in text.", "")
        if plan.get("citation_scope_contract"):
            system = system.replace(
                "Retain the core requirements and relevant limitations.",
                "Select findings that answer the research question and retain "
                "their relevant limitations.",
            ).replace(
                "Each unit must cite its actual premises through body_unit_refs.",
                "List the body units used as premises in body_unit_refs.",
            )
        from jsonschema import Draft202012Validator

        units = payload["body_units"]
        first = next(iter(units), None)
        mark = None
        for uid, premise in units.items():
            text = premise if isinstance(premise, str) else premise.get("text", "")
            if MARK.search(text):
                first, mark = (
                    uid,
                    references(
                        text, scope_contract=plan.get("citation_scope_contract")
                    )[0],
                )
                break
        example = {
            section: [
                {
                    "text": prose + (f"[C:{mark}]" if mark else "") + "。",
                    "body_unit_refs": [first],
                    "paragraph_start": True,
                }
            ]
            for section, prose in (
                ("abstract", "Abstract prose"),
                ("conclusion", "Conclusion prose"),
            )
        }
        Draft202012Validator(plan["wire_schema"]).validate(example)
        example_mark = f" such as [C:{mark}]" if mark else ""
        instructions = INSTRUCTION.replace(" such as [C:c3]", example_mark, 1)
        system += (
            "\n"
            + instructions
            + "\nJSON encoding example only: "
            + json.dumps(
                example, ensure_ascii=not bool(plan.get("citation_scope_contract"))
            )
        )
        system += (
            "\nGive every citation an adjacent clause with prose. Never append a "
            "group of source marks after the final punctuation of a unit. "
            "uN identifies a body unit, cN a source card: never change one prefix "
            "into the other. Copy marks only from the declared body_unit_refs."
        )
        if mark:
            system += (
                f"\nWrong placement: Conclusion prose。[C:{mark}]\n"
                "The host rejects marks placed after the clause's punctuation."
            )
        if plan.get("citation_scope_contract"):
            system += (
                "\nFor each scientific passage, carry over the finding and its "
                "marks from the same cited passage in the selected body units. "
                "A mark elsewhere in a selected unit cannot support a different "
                "passage. A synthesis needs each premise and its own marks. "
                "Report results with their stated conditions. Do not add claims "
                "about transfer, cross-task rankings or what a paper did not report."
            )
        elif GROUNDING_INSTRUCTION in original_system:
            system += GROUNDING_INSTRUCTION
    elif plan["kind"] == "repair" and system != repair_system(schema):
        invalid("writer_inline_repair_prompt_invalid")
    if plan.get("citation_scope_contract") and plan["kind"] != "repair":
        system = scope_instruction(system)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def cited_summary_probe(system, payload, schema, aliases):
    """Measure the longest example separately from the authored-text allowance."""
    from litagent.writing_v2 import request_messages

    units = payload["body_units"]
    if not aliases or not units:
        return None
    uid = next(reversed(units))
    mark = max(aliases, key=len)
    # All unit IDs and card numbers are ASCII. This is the largest legal
    # example for this unit count, and its added probe text is conservative.
    example_text = f"x[C:{mark}]。"
    example_unit = units[uid]
    if isinstance(example_unit, dict):
        example_unit = {**example_unit, "text": example_text}
    else:
        example_unit = example_text
    probe = {**payload, "body_units": {**units, uid: example_unit}}
    return request_messages(system, probe, schema, wire_contract="flat-writer-v1")


def validate_scopes(survey, bindings):
    """Require exact local ranges, owning units, evidence, and full text coverage."""
    from litagent.evidence import extract_evidence_refs_ordered

    scopes = bindings.get("citation_scopes")
    if bindings.get("citation_contract") != CONTRACT or not isinstance(scopes, list):
        invalid("writer_inline_citation_binding_invalid")
    by_unit = {}
    for scope in scopes:
        by_unit.setdefault(scope["unit_id"], []).append(scope)
    ordered = []
    for unit in bindings["units"]:
        start, end = unit["source_start"], unit["source_end"]
        actual = by_unit.pop(unit["unit_id"], [])
        expected = []
        for left, right, _ in _segments(
            survey[start:end],
            EVIDENCE,
            scope_contract=bindings.get("citation_scope_contract"),
        ):
            text = survey[start + left : start + right]
            a = start + left + len(text) - len(text.lstrip())
            b = start + left + len(text.rstrip())
            expected.append((a, b))
        if len(actual) != len(expected):
            invalid("writer_inline_citation_coverage_invalid")
        for index, (scope, (a, b)) in enumerate(zip(actual, expected, strict=True)):
            quote = survey[a:b]
            if (
                scope["scope_index"] != index
                or scope["source_start"] != a
                or scope["source_end"] != b
                or scope["source_quote"] != quote
                or scope["evidence_ids"] != extract_evidence_refs_ordered(quote)
                or scope["path"] != unit["path"]
            ):
                invalid(
                    "writer_inline_citation_binding_invalid", unit_id=unit["unit_id"]
                )
            ordered.append(scope)
    if by_unit or ordered != scopes:
        invalid("writer_inline_citation_binding_invalid")
    return scopes
