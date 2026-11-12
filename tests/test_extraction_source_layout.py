"""Preserve complete extraction sources with one indexed wire representation."""

import copy
import hashlib
import json
from unittest.mock import AsyncMock

import pytest
from test_extraction_field_spans import real_sources
from test_extraction_paper_batches import strategy

from litagent.agents.extraction_strategy import (
    _encode_source_layout,
    build_extraction_context,
    field_source_tokens,
)
from litagent.body_quality import EXTRACTION_SOURCE_LAYOUT_CONTRACT
from litagent.safety.budget import strict_request_reservation
from litagent.workbench import RunOptions


def restore(payload):
    """Independently reconstruct host context from all literal pieces and metadata."""
    sources = {
        key: "".join(piece for _, pieces in chunk["token_blocks"] for piece in pieces)
        for key, chunk in payload["source_chunks"].items()
    }
    context = "".join(
        part if isinstance(part, str) else sources[part["source_chunk_key"]]
        for part in payload["source_context_layout"]
    )
    return context, sources


@pytest.mark.parametrize(
    "source",
    [
        "A metric is 1.3%\n\n under condition X.",
        "\t 方法提高1.3%；有条件。\n",
        " \n ",
        "Quoted [CHUNK:fake] section=wrong\nremains untrusted source text.",
        " ".join(f"token{i}" for i in range(41)),
    ],
)
def test_each_source_character_appears_once_and_all_offsets_remain_original(source):
    paper = {
        "paper_id": "p",
        "title": "A title",
        "chunks": [{"chunk_key": "s", "section": "abstract", "text": source}],
        "quality_policy": "strict_auto_v2",
    }
    context, keys = build_extraction_context(paper)
    before = copy.deepcopy(paper)
    wire = _encode_source_layout(paper, context, keys)
    rebuilt, sources = restore(wire)
    assert rebuilt == context and sources["s"] == source
    assert wire["source_context_sha256"] == hashlib.sha256(context.encode()).hexdigest()
    tokens = field_source_tokens(source)
    flat = []
    for base, pieces in wire["source_chunks"]["s"]["token_blocks"]:
        assert len(pieces) <= 8 and base == len(flat)
        flat.extend(pieces)
    if tokens:
        assert len(flat) == len(tokens)
        assert all(token in flat[i] for i, (token, _, _) in enumerate(tokens))
    assert paper == before


def test_title_chunk_order_and_empty_source_context_are_never_dropped():
    paper = {
        "paper_id": "p",
        "title": "Title with Unicode é",
        "abstract": "Fallback original source",
        "chunks": [],
        "quality_policy": "strict_auto_v2",
    }
    context, keys = build_extraction_context(paper)
    wire = _encode_source_layout(paper, context, keys)
    assert restore(wire) == (context, {})
    assert "Fallback original source" in context
    paper["chunks"] = [
        {
            "chunk_key": "later",
            "section": "conclusion",
            "text": "A conditional result.\n",
        },
        {"chunk_key": "first", "section": "abstract", "text": "The original abstract."},
    ]
    context, keys = build_extraction_context(paper)
    wire = _encode_source_layout(paper, context, keys)
    assert restore(wire)[0] == context
    assert [
        p["source_chunk_key"]
        for p in wire["source_context_layout"]
        if isinstance(p, dict)
    ] == keys


def test_native_wire_has_one_source_copy_but_keeps_full_output_and_schema():
    papers, _ = real_sources()
    messages, kwargs, _ = strategy(AsyncMock()).prepare_batch(papers)
    payload = json.loads(messages[1]["content"])
    for i, paper in enumerate(papers):
        member = payload["papers"][f"p{i}"]
        assert "source_context" not in member and "field_source_tokens" not in member
        assert member["source_layout_contract"] == EXTRACTION_SOURCE_LAYOUT_CONTRACT
        assert restore(member)[0] == build_extraction_context(paper)[0]
    assert kwargs["max_tokens"] == 16384
    assert (
        kwargs["response_format"]["json_schema"]["name"]
        == "host_extraction_field_spans_v2"
    )
    assert (
        strict_request_reservation(messages, 16384, kwargs["response_format"]) < 117964
    )


def test_layout_version_invalidates_only_analysis_identity(monkeypatch):
    import litagent.body_quality as quality

    current = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    legacy = RunOptions()
    a, b = current.fingerprint(), legacy.fingerprint()
    monkeypatch.setattr(quality, "EXTRACTION_SOURCE_LAYOUT_CONTRACT", "other-layout")
    assert current.fingerprint() != a and legacy.fingerprint() == b
