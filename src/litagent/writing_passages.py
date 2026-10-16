"""Compile source selections without asking a model to copy original paper text."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping

from litagent.body_quality import ANALYSIS_VERSION

VERSION = "grounded-writing-v18"
CONTRACT = "host-writing-passage-selection-v1"
FULL_VERSION = "grounded-writing-v20"
FULL_CONTRACT = "host-writing-full-sources-v1"
NOTICE_VERSION = "grounded-writing-v21"
SCOPE_NOTICE_CONTRACT = "scope-notice-v1"
BODY_SECTION_KEYS = (
    "introduction",
    "taxonomy",
    "methods",
    "experiments",
    "open_problems",
)


def scope_notice_catalog(language: str) -> dict[str, str]:
    """Return report-scope statements without adding empirical or field-wide claims."""
    if language == "zh-CN":
        return {
            "review_scope": "本综述围绕给定问题组织所选文献证据。",
            "evidence_limit": "本综述不据所选材料的缺失推断整个研究领域不存在相关工作。",
        }
    if language == "en":
        return {
            "review_scope": "This review organizes the selected literature evidence around the stated question.",
            "evidence_limit": "This review does not infer a field-wide absence of research from missing selected material.",
        }
    raise ValueError("unsupported_report_language")


def expand_scope_notice(unit: dict, language: str) -> dict:
    """Reject free non-scientific prose instead of silently discarding its premises."""
    if not isinstance(unit, dict) or set(unit) != {
        "text",
        "kind",
        "fact_card_ids",
        "scope_notice_id",
    }:
        raise ValueError("strict_writing_scope_notice_invalid")
    result = copy.deepcopy(unit)
    notice = result.pop("scope_notice_id")
    if not isinstance(notice, str):
        raise ValueError("strict_writing_scope_notice_invalid")
    if result["kind"] == "scientific":
        if notice != "none":
            raise ValueError("strict_writing_scope_notice_invalid")
    elif result["kind"] in {"organization", "gap"}:
        if (
            result["text"] != ""
            or result["fact_card_ids"] != []
            or notice not in scope_notice_catalog(language)
        ):
            raise ValueError("strict_writing_scope_notice_invalid")
        result["text"] = scope_notice_catalog(language)[notice]
    else:
        raise ValueError("strict_writing_scope_notice_invalid")
    return result


def bind_scope_notices(survey: str, notices: Mapping, language: str) -> dict:
    """Bind compiler-owned notices to exact rendered occurrences in their section."""
    from litagent.review_structure import structure_sections, top_level_sections

    catalog = scope_notice_catalog(language)
    titles = dict(structure_sections("academic_v2", language))
    sections = top_level_sections(survey)
    if not isinstance(notices, Mapping):
        raise ValueError("strict_writing_scope_notice_binding_invalid")
    result, used = {}, set()
    for location, row in notices.items():
        if (
            not isinstance(location, str)
            or not re.fullmatch(r"[a-z_]+\.blocks\.\d+\.sentences\.\d+", location)
            or not isinstance(row, Mapping)
        ):
            raise ValueError("strict_writing_scope_notice_binding_invalid")
        if row.get("native_location", location) != location:
            raise ValueError("strict_writing_scope_notice_binding_invalid")
        key = location.split(".")[0]
        notice_id = row.get("notice_id")
        if (
            not isinstance(notice_id, str)
            or notice_id not in catalog
            or row.get("text") != catalog[notice_id]
            or row.get("contract") != SCOPE_NOTICE_CONTRACT
        ):
            raise ValueError("strict_writing_scope_notice_binding_invalid")
        owners = [
            section for section in sections if section["section"] == titles.get(key)
        ]
        if len(owners) != 1:
            raise ValueError("strict_writing_scope_notice_binding_invalid")
        owner, text = owners[0], row["text"]
        positions = [
            m.start() + owner["content_start"]
            for m in re.finditer(
                re.escape(text), survey[owner["content_start"] : owner["end"]]
            )
        ]
        if len(positions) != 1 or (key, notice_id) in used:
            raise ValueError("strict_writing_scope_notice_binding_invalid")
        used.add((key, notice_id))
        result[location] = {
            "notice_id": notice_id,
            "text": text,
            "contract": SCOPE_NOTICE_CONTRACT,
            "native_location": location,
            "source_start": positions[0],
            "source_end": positions[0] + len(text),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    expected = set()
    for owner in sections:
        if owner["section"] not in titles.values():
            continue
        for text in catalog.values():
            for match in re.finditer(
                re.escape(text), survey[owner["content_start"] : owner["end"]]
            ):
                start = owner["content_start"] + match.start()
                expected.add((start, start + len(text)))
    if {
        (row["source_start"], row["source_end"]) for row in result.values()
    } != expected:
        raise ValueError("strict_writing_scope_notice_binding_invalid")
    return result


def validate_scope_notice_bindings(
    survey: str, bindings: Mapping, language: str
) -> None:
    """Require exact report identity, rendered positions and compiler notice metadata."""
    if (
        not isinstance(bindings, Mapping)
        or bindings.get("scope_notice_contract") != SCOPE_NOTICE_CONTRACT
        or bindings.get("report_sha256") != hashlib.sha256(survey.encode()).hexdigest()
        or "scope_notices" not in bindings
    ):
        raise ValueError("strict_writing_scope_notice_binding_invalid")
    notices = bindings["scope_notices"]
    if bind_scope_notices(survey, notices, language) != notices:
        raise ValueError("strict_writing_scope_notice_binding_invalid")


def has_research_gap_predicate(text: str) -> bool:
    """Detect explicit claims about missing research, rather than missing inputs."""
    return bool(
        re.search(
            r"(?:现有研究|研究|方法|能力|表现).{0,45}(?:缺乏.{0,12}(?:比较|对比|分析)|"
            r"尚未.{0,14}验证|未得到.{0,10}验证)|"
            r"\b(?:research|literature|studies)\b.{0,45}\b(?:lacks?|lack of|no systematic)\b|"
            r"\b(?:generalization|performance|methods?)\b.{0,45}"
            r"\b(?:not (?:yet |been |sufficiently )*validated|remain\w* unvalidated)\b",
            text,
            re.I,
        )
    )


def summary_binding_schema(body_sections) -> dict:
    """Describe explicit selectors without claiming dynamic targets exist."""
    sections = list(body_sections)
    if any(key not in BODY_SECTION_KEYS for key in sections):
        raise ValueError("strict_writing_summary_binding_invalid")
    properties = {
        "section_key": {
            "type": "string",
            "enum": sections,
            "description": "Choose one real body unit. Its complete text, kind and citations are derived by the host.",
        },
        "block_index": {
            "type": "integer",
            "description": "Zero selects first_block; one selects additional_blocks[0], etc.",
        },
        "sentence_index": {
            "type": "integer",
            "description": "Zero selects first_sentence; one selects additional_sentences[0], etc.",
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _body_reference(reference, *, scientific: bool) -> str:
    """Build one exact host location; never repair aliases or coerce indices."""
    if (
        not isinstance(reference, dict)
        or set(reference) != {"section_key", "block_index", "sentence_index"}
        or not isinstance(reference.get("section_key"), str)
        or type(reference.get("block_index")) is not int
        or type(reference.get("sentence_index")) is not int
        or reference["block_index"] < 0
        or reference["sentence_index"] < 0
    ):
        raise ValueError("strict_writing_summary_binding_invalid")
    section, block, sentence = (
        reference["section_key"],
        reference["block_index"],
        reference["sentence_index"],
    )
    if section in BODY_SECTION_KEYS:
        return f"{section}.blocks.{block}.sentences.{sentence}"
    if scientific or section != "" or block != 0 or sentence != 0:
        raise ValueError("strict_writing_summary_binding_invalid")
    return ""


def build_passage_catalog(cards: Mapping) -> dict:
    """Partition every original byte once, keeping stable source and card ownership."""
    from litagent.eval.scoped import _source_boundary_tokens
    from litagent.quality import FACT_VERSION, digest

    sources, card_sources, identities = {}, {}, {}
    for cid, card in cards.items():
        text = card.get("source_quote")
        if (
            not isinstance(text, str)
            or not text.strip()
            or card.get("card_id") != cid
            or hashlib.sha256(text.encode()).hexdigest() != card.get("content_sha256")
            or card.get("source_start") != 0
            or card.get("source_end") != len(text)
            or cid
            != "f:"
            + digest(
                [
                    FACT_VERSION,
                    card.get("paper_id"),
                    card.get("evidence_id"),
                    card.get("content_sha256"),
                    card.get("chunk_key"),
                    0,
                    len(text),
                ]
            )[:24]
        ):
            raise ValueError("strict_writing_passage_source_invalid")
        key = digest([card["paper_id"], card["content_sha256"]])
        sid = identities.setdefault(key, f"r{len(identities) + 1}")
        card_sources[cid] = sid
        if sid in sources:
            sources[sid]["fact_card_ids"].append(cid)
            continue
        ends = sorted(
            {
                p + 1
                for p, token in _source_boundary_tokens(text)
                if token in {".", "!", "?", "。", "！", "？"}
            }
            | {len(text)}
        )
        parts, start = [], 0
        for end in ends:
            if end <= start:
                continue
            parts.append(
                {
                    "passage_id": f"{sid}.p{len(parts) + 1}",
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
            start = end
        sources[sid] = {
            "paper_id": card["paper_id"],
            "source_sha256": card["content_sha256"],
            "fact_card_ids": [cid],
            "full_passage_id": sid + ".full",
            "parts": parts,
        }
    return {"contract": CONTRACT, "card_sources": card_sources, "sources": sources}


def select_passage(card_id: str, passage_id: str, catalog: Mapping) -> dict:
    """Resolve one same-source continuous interval; never infer scientific entailment."""
    sid = catalog["card_sources"].get(card_id)
    match = re.fullmatch(
        r"(r\d+)\.(?:full|p([1-9]\d*)(?:-(r\d+)\.p([1-9]\d*))?)", passage_id
    )
    if not sid or not match or match[1] != sid or (match[3] and match[3] != sid):
        raise ValueError("strict_writing_passage_binding_invalid")
    source = catalog["sources"][sid]
    parts = source["parts"]
    if match[2] is None:
        chosen = parts
    else:
        first, last = int(match[2]), int(match[4] or match[2])
        if not 1 <= first <= last <= len(parts):
            raise ValueError("strict_writing_passage_range_invalid")
        chosen = parts[first - 1 : last]
    quote = "".join(part["text"] for part in chosen)
    if not quote.strip():
        raise ValueError("strict_writing_passage_empty")
    return {
        "passage_id": passage_id,
        "fact_card_id": card_id,
        "source_quote": quote,
        "quote_start": chosen[0]["start"],
        "quote_end": chosen[-1]["end"],
        "source_sha256": source["source_sha256"],
        "verification": "source_location_only",
    }


def _unique(pairs):
    """Preserve duplicate-key rejection before normalizing a native candidate."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("strict_writing_duplicate_object_key")
        result[key] = value
    return result


def compile_passage_report(
    content: str,
    cards: Mapping,
    *,
    binding_mode: str = "passages",
    require_scope_notices: bool = False,
    language: str | None = None,
    notice_bindings: dict | None = None,
    analysis_mode: bool = False,
    analysis_bindings: dict | None = None,
) -> tuple[str, dict]:
    """Derive canonical summary citations from real scientific body units only."""
    body_notice_ids = set()
    from litagent.quality import GroundedWritingError

    if binding_mode not in {"passages", "full_sources"}:
        raise ValueError("strict_writing_binding_mode_invalid")
    full_sources = binding_mode == "full_sources"
    if require_scope_notices and (not full_sources or language not in {"en", "zh-CN"}):
        raise ValueError("strict_writing_scope_notice_invalid")
    try:
        document = json.loads(content, object_pairs_hook=_unique)
    except json.JSONDecodeError as exc:
        raise ValueError("strict_writing_native_contract_invalid") from exc
    if not isinstance(document, dict) or not isinstance(document.get("sections"), dict):
        raise ValueError("strict_writing_native_contract_invalid")
    canonical = copy.deepcopy(document)
    catalog, spans, body, errors = build_passage_catalog(cards), {}, {}, []

    def visit(section):
        blocks = [section["first_block"], *section["additional_blocks"]]
        for bi, block in enumerate(blocks):
            for si, sentence in enumerate(
                [block["first_sentence"], *block["additional_sentences"]]
            ):
                yield bi, si, sentence

    def record(key, bi, si, sentence, exc):
        original_section = document["sections"][key]
        original_block = (
            original_section["first_block"]
            if bi == 0
            else original_section["additional_blocks"][bi - 1]
        )
        original_sentence = (
            original_block["first_sentence"]
            if si == 0
            else original_block["additional_sentences"][si - 1]
        )
        errors.append(
            {
                "reason_code": str(exc),
                "location": f"{key}.blocks.{bi}.sentences.{si}",
                "section_key": key,
                "candidate_text": sentence.get("text"),
                "actual_fact_card_ids": original_sentence.get("fact_card_ids", []),
                **(
                    {
                        "actual_body_reference": copy.deepcopy(
                            original_sentence.get("body_reference")
                        ),
                        **(
                            {
                                "actual_body_references": copy.deepcopy(
                                    original_sentence["body_references"]
                                )
                            }
                            if "body_references" in original_sentence
                            else {}
                        ),
                    }
                    if key in {"abstract", "conclusion"}
                    else {}
                ),
                **(
                    {"actual_body_sentence_id": original_sentence["body_sentence_id"]}
                    if "body_sentence_id" in original_sentence
                    else {}
                ),
                "raw_json_path": f"sections.{key}."
                + ("first_block" if bi == 0 else f"additional_blocks.{bi-1}")
                + "."
                + ("first_sentence" if si == 0 else f"additional_sentences.{si-1}"),
            }
        )

    try:
        for key, section in canonical["sections"].items():
            if key in {"abstract", "conclusion"}:
                continue
            for bi, si, sentence in visit(section):
                location = f"{key}.blocks.{bi}.sentences.{si}"
                try:
                    expected = {"kind", "fact_card_ids", "text"}
                    if require_scope_notices:
                        expanded = expand_scope_notice(sentence, language)
                        if sentence["scope_notice_id"] != "none":
                            if sentence["scope_notice_id"] in body_notice_ids:
                                raise ValueError("strict_writing_scope_notice_repeated")
                            body_notice_ids.add(sentence["scope_notice_id"])
                            if notice_bindings is not None:
                                notice_bindings[location] = {
                                    "notice_id": sentence["scope_notice_id"],
                                    "text": expanded["text"],
                                    "contract": SCOPE_NOTICE_CONTRACT,
                                }
                        sentence.clear()
                        sentence.update(expanded)
                    if not full_sources:
                        expected.add("support_passage_ids")
                    if set(sentence) != expected:
                        raise ValueError("strict_writing_passage_fields_invalid")
                    ids = sentence["fact_card_ids"]
                    if (
                        not isinstance(ids, list)
                        or any(not isinstance(cid, str) for cid in ids)
                        or len(ids) != len(set(ids))
                    ):
                        raise ValueError("strict_writing_passage_binding_invalid")
                    if full_sources:
                        if sentence[
                            "kind"
                        ] != "scientific" and has_research_gap_predicate(
                            sentence["text"]
                        ):
                            raise ValueError("strict_writing_uncited_scientific_gap")
                        if (sentence["kind"] == "scientific" and not ids) or (
                            sentence["kind"] != "scientific" and ids
                        ):
                            raise ValueError("strict_writing_passage_binding_invalid")
                        # The complete source interval is host-owned, not a semantic verdict.
                        selected = [
                            catalog["card_sources"].get(cid, "") + ".full"
                            for cid in ids
                        ]
                    else:
                        selected = sentence["support_passage_ids"]
                    if (
                        not isinstance(selected, list)
                        or len(ids) != len(selected)
                        or any(not isinstance(pid, str) for pid in selected)
                    ):
                        raise ValueError("strict_writing_passage_binding_invalid")
                    located = [
                        select_passage(cid, pid, catalog)
                        for cid, pid in zip(ids, selected)
                    ]
                    sentence["support_quotes"] = [
                        item["source_quote"] for item in located
                    ]
                    if not full_sources:
                        del sentence["support_passage_ids"]
                    spans[location] = located
                    body[location] = (
                        copy.deepcopy(sentence),
                        copy.deepcopy(located),
                    )
                except (ValueError, TypeError) as exc:
                    record(key, bi, si, sentence, exc)
        for key in ("abstract", "conclusion"):
            if key not in canonical["sections"]:
                continue
            selected_body_units = set()
            selected_texts = set()
            for bi, si, sentence in visit(canonical["sections"][key]):
                location = f"{key}.blocks.{bi}.sentences.{si}"
                try:
                    if analysis_mode:
                        if (
                            set(sentence) != {"text", "body_references"}
                            or not isinstance(sentence["text"], str)
                            or not sentence["text"].strip()
                        ):
                            raise ValueError("strict_writing_summary_binding_invalid")
                        refs = sentence["body_references"]
                        if not isinstance(refs, list) or not 1 <= len(refs) <= 6:
                            raise ValueError("strict_writing_summary_binding_invalid")
                        bids = [_body_reference(ref, scientific=True) for ref in refs]
                        if len(set(bids)) != len(bids) or any(
                            bid not in body or body[bid][0]["kind"] != "scientific"
                            for bid in bids
                        ):
                            raise ValueError("strict_writing_summary_binding_invalid")
                        text = sentence["text"].strip()
                        if text in selected_texts:
                            raise ValueError("strict_writing_summary_content_duplicate")
                        selected_texts.add(text)
                        located_by_card = {
                            item["fact_card_id"]: copy.deepcopy(item)
                            for bid in bids
                            for item in body[bid][1]
                        }
                        if not located_by_card:
                            raise ValueError("strict_writing_summary_binding_invalid")
                        located = list(located_by_card.values())
                        sentence.clear()
                        sentence.update(
                            text=text,
                            kind="scientific",
                            fact_card_ids=list(located_by_card),
                            support_quotes=[item["source_quote"] for item in located],
                        )
                        spans[location] = located
                        if analysis_bindings is not None:
                            analysis_bindings[location] = {
                                "body_locations": bids,
                                "text_sha256": hashlib.sha256(
                                    text.encode()
                                ).hexdigest(),
                            }
                        continue
                    if set(sentence) != {"body_reference"}:
                        raise ValueError("strict_writing_summary_binding_invalid")
                    bid = _body_reference(
                        sentence.pop("body_reference"),
                        scientific=True,
                    )
                    if bid not in body or bid in selected_body_units:
                        raise ValueError("strict_writing_summary_binding_invalid")
                    selected_body_units.add(bid)
                    body_sentence, located = body[bid]
                    if (
                        require_scope_notices
                        and notice_bindings is not None
                        and bid in notice_bindings
                    ):
                        notice_bindings[location] = copy.deepcopy(notice_bindings[bid])
                    content_key = re.sub(r"\s+", " ", body_sentence["text"]).strip()
                    if full_sources and content_key in selected_texts:
                        raise ValueError("strict_writing_summary_content_duplicate")
                    selected_texts.add(content_key)
                    sentence.update(copy.deepcopy(body_sentence))
                    if sentence["kind"] == "scientific":
                        if (
                            body_sentence["kind"] != "scientific"
                            or not body_sentence["fact_card_ids"]
                            or not located
                        ):
                            raise ValueError("strict_writing_summary_binding_invalid")
                        sentence["fact_card_ids"] = copy.deepcopy(
                            body_sentence["fact_card_ids"]
                        )
                        sentence["support_quotes"] = copy.deepcopy(
                            body_sentence["support_quotes"]
                        )
                        spans[location] = copy.deepcopy(located)
                    else:
                        if body_sentence["fact_card_ids"] or located:
                            raise ValueError("strict_writing_summary_binding_invalid")
                        sentence.update(fact_card_ids=[], support_quotes=[])
                        spans[location] = []
                except (ValueError, TypeError) as exc:
                    record(key, bi, si, sentence, exc)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("strict_writing_native_contract_invalid") from exc
    if errors:
        raise GroundedWritingError(content, errors)
    if require_scope_notices:
        # Scope notices cannot replace scientific limitations or all summary findings.
        for key in ("open_problems", "abstract", "conclusion"):
            if key in canonical["sections"] and not any(
                s.get("kind") == "scientific"
                for _, _, s in visit(canonical["sections"][key])
            ):
                raise ValueError("strict_writing_scientific_section_missing")
    return json.dumps(canonical, ensure_ascii=False), spans
