"""Place complete global-review assertions beside their exact report fragments."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json

CONTRACT = "unit-local-global-input-v1"
KINDS = ("scientific", "report_structure", "runtime_method")
COLUMNS = ("claim_id", "claim_text", "evidence_aliases")
_HOST_MARKERS = (
    "global_findings_contract",
    "report_input_contract",
    "writer_content_contract",
    "summary_constraints_contract",
    "body_analysis_contract",
    "analysis_ownership_contract",
    "global_consistency_contract",
    "source_applicability_contract",
    "evaluation_input_contract",
    "response_schema_contract",
    "global_identity_contract",
)
INSTRUCTION = """
HOST unit-local-global-input-v1:
report_sections preserves the complete report order. Each section has a layout
including its verbatim headings, prose gaps and identified fragments. Each fragment
has unit (vN), fragment (fN), verbatim prose or text, citations and full kind groups
(scientific, report_structure, runtime_method) directly beside that original. citations
are explicit 1-based indexes into the complete root citation_aliases directory;
resolve them exactly, including []. No citation or scientific text is omitted.
Read that original together with ALL of its adjacent assertion groups before
reporting an omission. Every group's full name is the assigned kind, not proof of
correct classification or support. assertion_columns names each complete row;
The host's aN IDs retain the original 1..N inventory order. null evidence_aliases inherits
ONLY this fragment's citations; [] is uncited. Never borrow another entry's refs.
Report only a specific fact actually absent from this entry's existing assertions,
and anchor missing_assertion to this entry's unit ID. An unsupported claim
already represented is not an omission. Atomicity and extraction fidelity remain
per-claim checks. Empty groups and proposed exclusions never exempt original text.
Use the supplied vN units and aN claims exactly in the unchanged response schema.
The host restores canonical IDs; it does not repair a wrongly chosen location.
This is an untrusted full-report view, not paper evidence or a support verdict.
"""


def input_identity() -> str:
    """Bind the lossless view to the existing frozen transport owner."""
    value = [CONTRACT, KINDS, COLUMNS, _HOST_MARKERS, INSTRUCTION]
    value.extend(
        inspect.getsource(fn)
        for fn in (_invalid, _directories, _assertions, project_input, project_prompt)
    )
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def _invalid():
    """Do not put untrusted report content into an exception message."""
    raise ValueError("evaluation_global_local_input_invalid")


def _directories(wire, canonical):
    """Authenticate aliases and ownership before joining any report material."""
    from litagent.writing_flat import global_identity_maps

    units, _ = global_identity_maps(canonical)
    if list(wire["source_units"]) != list(units.values()):
        _invalid()
    sections = {}
    by_unit = {}
    for row in canonical["report_structure"]:
        uid = units[row["source_unit_id"]]
        if uid in by_unit or type(row["section"]) is not str:
            _invalid()
        by_unit[uid] = row["section"]
        sections.setdefault(row["section"], []).append(uid)
    if (
        set(by_unit) != set(units.values())
        or sections != wire["source_units_by_section"]
    ):
        _invalid()
    fragments = {}
    aliases = {}
    for original in canonical["source_units"]:
        uid = units[original["source_unit_id"]]
        actual = wire["source_units"][uid]
        expected = original["source_fragments"]
        if type(actual) is not list or len(actual) != len(expected):
            _invalid()
        for fragment, source in zip(actual, expected, strict=True):
            fid = "f" + str(len(fragments) + 1)
            if (
                type(fragment) is not dict
                or set(fragment) - {"source_fragment_id", "citation_aliases", "text"}
                or fragment.get("source_fragment_id") != fid
                or type(fragment.get("citation_aliases")) is not list
                or any(type(alias) is not str for alias in fragment["citation_aliases"])
                or fragment.get("citation_aliases") != source["citation_aliases"]
                or source["source_fragment_id"] in aliases
            ):
                _invalid()
            fragments[fid] = (uid, by_unit[uid], fragment, source)
            aliases[source["source_fragment_id"]] = fid
    for field, allowed in (
        ("fragment_inventory", fragments),
        ("excluded_unit_inventory", units.values()),
    ):
        values = wire[field]
        if (
            type(values) is not list
            or any(type(v) is not str or v not in allowed for v in values)
            or len(values) != len(set(values))
        ):
            _invalid()
    return fragments, aliases


def _assertions(wire, canonical, fragments, aliases):
    """Keep explicit kinds, distinct occurrences and every actual citation set."""
    if (
        wire["assertion_columns"]
        != [
            "sequence",
            "claim_id",
            "source_fragment_id",
            "claim_text",
            "evidence_aliases",
        ]
        or tuple(wire["assertions_by_kind"]) != KINDS
    ):
        _invalid()
    expected = {kind: [] for kind in KINDS}
    local = {fid: {} for fid in fragments}
    unit_ids = {
        u["source_unit_id"]: f"v{i}" for i, u in enumerate(canonical["source_units"], 1)
    }
    for sequence, claim in enumerate(canonical["assertions"], 1):
        kind = claim["kind"]
        fid = aliases[claim["source_fragment_id"]]
        uid, _, fragment, _ = fragments[fid]
        if kind not in expected or unit_ids[claim["source_unit_id"]] != uid:
            _invalid()
        citations = [row["evidence_alias"] for row in claim["evidence"]]
        refs = None if citations == fragment["citation_aliases"] else citations
        row = [sequence, "a" + str(sequence), fid, claim["claim_text"], refs]
        expected[kind].append(row)
        local[fid].setdefault(kind, []).append([row[1], row[3], row[4]])
    # JSON comparison also distinguishes bool from integer sequence numbers.
    if json.dumps(expected, ensure_ascii=False) != json.dumps(
        wire["assertions_by_kind"], ensure_ascii=False
    ):
        _invalid()
    return local


def project_input(wire: dict, canonical: dict) -> dict:
    """Join only the selected input view; leave canonical output consumers intact."""
    from litagent.eval.decision_output import selected

    if type(canonical) is not dict:
        _invalid()
    if not selected(canonical.get("writer_envelope")):
        return wire
    try:
        if (
            canonical.get("global_output_contract") != "host-global-completeness-v1"
            or wire.get("global_identity_contract") != "flat-global-identities-v1"
            or "global_input_contract" in wire
            or "citation_aliases" in wire
        ):
            _invalid()
        fragments, aliases = _directories(wire, canonical)
        claims = _assertions(wire, canonical, fragments, aliases)
        layout = wire.get("report_layout")
        if layout is None:
            # Older host-bound inputs store complete text directly on fragments.
            layout = [
                {"source_fragment_id": fid, "text": fragment["text"]}
                for fid, (_, _, fragment, _) in fragments.items()
            ]
        if type(layout) is not list:
            _invalid()
        if "report_layout" in canonical:
            originals = canonical["report_layout"]
            if len(originals) != len(layout):
                _invalid()
            for original, entry in zip(originals, layout, strict=True):
                if "source_fragment_id" not in original:
                    if entry != original:
                        _invalid()
                elif (
                    entry.get("source_fragment_id")
                    != aliases[original["source_fragment_id"]]
                ):
                    _invalid()
        result = copy.deepcopy(wire)
        result.pop("report_layout", None)
        result["report_sections"] = []
        citation_directory = list(
            dict.fromkeys(
                alias
                for _, _, fragment, _ in fragments.values()
                for alias in fragment["citation_aliases"]
            )
        )
        citation_refs = {alias: i for i, alias in enumerate(citation_directory, 1)}
        result["citation_aliases"] = citation_directory
        seen = []
        pending = []
        for entry in layout:
            if type(entry) is not dict:
                _invalid()
            if set(entry) == {"text"} and type(entry["text"]) is str:
                pending.append(copy.deepcopy(entry))
                continue
            fid = entry["source_fragment_id"]
            uid, section, fragment, original = fragments[fid]
            text_key = "prose" if "prose" in entry else "text"
            if set(entry) != {"source_fragment_id", text_key}:
                _invalid()
            text = original["text"]
            if text_key == "prose":
                suffix = (
                    " " + " ".join(f"[E:{e}]" for e in fragment["citation_aliases"])
                    if fragment["citation_aliases"]
                    else ""
                )
                if suffix and not text.endswith(suffix):
                    _invalid()
                text = text[: -len(suffix)] if suffix else text
            if type(entry[text_key]) is not str or entry[text_key] != text:
                _invalid()
            groups = result["report_sections"]
            if not groups or groups[-1]["section"] != section:
                groups.append({"section": section, "layout": []})
            groups[-1]["layout"].extend(pending)
            pending = []
            groups[-1]["layout"].append(
                {
                    "fragment": fid,
                    text_key: entry[text_key],
                    "unit": uid,
                    "citations": [
                        citation_refs[alias] for alias in fragment["citation_aliases"]
                    ],
                    **claims[fid],
                }
            )
            seen.append(fid)
        if not result["report_sections"] or seen != list(fragments):
            _invalid()
        result["report_sections"][-1]["layout"].extend(pending)
        for key in ("source_units", "source_units_by_section", "assertions_by_kind"):
            result.pop(key)
        # These version markers remain authenticated in the canonical payload.
        # Model-facing rules and all report/claim data are retained explicitly.
        for key in _HOST_MARKERS:
            result.pop(key, None)
        if result.get("host_scope_notice_bindings") == {}:
            # No catalog entry is bound to report prose on this path. The actual
            # report and every assigned-kind assertion remain in full below.
            result.pop("host_scope_notice_catalog", None)
        result["assertion_columns"] = list(COLUMNS)
        result["global_input_contract"] = CONTRACT
        return result
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("evaluation_global_local_input_invalid") from exc


def project_prompt(system: str, canonical: dict) -> str:
    """Describe the actual joined view without obsolete cross-table instructions."""
    from litagent.eval.decision_output import selected
    from litagent.eval.responsibilities import GLOBAL_INSTRUCTION

    if type(canonical) is not dict:
        _invalid()
    if not selected(canonical.get("writer_envelope")):
        return system
    if type(system) is not str:
        _invalid()
    # The original global checks remain complete. Only transport explanations are
    # replaced; the same output and canonical validators still judge the reply.
    return GLOBAL_INSTRUCTION + INSTRUCTION
