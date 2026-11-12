"""Tests for skill discovery, ranking, and loading."""

import pytest

from litagent.skills.manager import SkillManager
from litagent.tools.worker_tools import make_load_skill_tool


class _OfflineSkillEmbedder:
    """Deterministic embeddings for ranking tests; never downloads a model."""

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        normalized = text.lower()
        if "writing a literature survey" in normalized:
            return [1.0, 0.0]
        if "reviewing" in normalized and "critiquing" in normalized:
            return [0.0, 1.0]
        return [0.0, 0.0]

    def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(texts, str):
            return self._embed_one(texts)
        return [self._embed_one(text) for text in texts]


@pytest.fixture(autouse=True)
def offline_skill_embedder(monkeypatch):
    monkeypatch.setattr(
        "litagent.skills.manager.get_embedder",
        lambda: _OfflineSkillEmbedder(),
    )


def _mgr():
    return SkillManager()


def test_scans_flat_all_skills():
    """Skill discovery scans the flat skill directory."""
    names = _mgr().list_names()
    assert {"cv", "nlp", "survey_writing", "review_checklist"} <= set(names)


def test_semantic_search_picks_relevant():
    """Semantic ranking selects the relevant skill."""
    m = _mgr()
    top = m.search_skills("writing a literature survey", top_k=2)
    names = [s.name for s in top]
    assert "survey_writing" in names


def test_semantic_search_review():
    m = _mgr()
    top = m.search_skills("reviewing and critiquing a draft", top_k=2)
    assert "review_checklist" in [s.name for s in top]


def test_metadata_for_filters_topk():
    """Metadata filtering respects the result limit."""
    m = _mgr()
    text = m.to_metadata_text_for("writing a literature survey", top_k=2)
    assert text.count("<skill") == 2


def test_search_all_when_fewer_than_topk():
    """Search returns every skill when below the limit."""
    m = _mgr()
    top = m.search_skills("anything", top_k=99)
    assert len(top) == len(m.list_names())


def test_metadata_for_degrades_to_full_on_error(monkeypatch):
    """Ranking failures degrade to the complete skill list."""
    m = _mgr()

    def _boom(*a, **k):
        raise RuntimeError("embedder OOM")

    monkeypatch.setattr(m, "search_skills", _boom)
    text = m.to_metadata_text_for("writing a survey", top_k=2)
    assert text.count("<skill") == len(m.list_names())


@pytest.mark.asyncio
async def test_load_skill_returns_body():
    m = _mgr()
    tool = make_load_skill_tool(m)
    body = await tool.ainvoke({"name": "survey_writing"})
    assert "Survey Writing Methodology" in body


@pytest.mark.asyncio
async def test_load_skill_unknown():
    tool = make_load_skill_tool(_mgr())
    assert "not found" in await tool.ainvoke({"name": "nope"})


@pytest.mark.asyncio
async def test_load_skill_none_manager():
    """Skill loading degrades safely without a manager."""
    tool = make_load_skill_tool(None)
    assert "not available" in await tool.ainvoke({"name": "cv"})
