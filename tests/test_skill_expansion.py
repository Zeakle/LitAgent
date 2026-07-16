import pytest
from litagent.skills.manager import SkillManager
from litagent.tools.worker_tools import make_load_skill_tool


def _mgr():
    return SkillManager()   # 默认扫 src/litagent/skills（扁平全部）


def test_scans_flat_all_skills():
    """扁平扫描加载全部 4 个 skill。"""
    names = _mgr().list_names()
    assert {"cv", "nlp", "survey_writing", "review_checklist"} <= set(names)


def test_semantic_search_picks_relevant():
    """语义检索：'writing a survey' → survey_writing 排最相关。"""
    m = _mgr()
    top = m.search_skills("writing a literature survey", top_k=2)
    names = [s.name for s in top]
    assert "survey_writing" in names


def test_semantic_search_review():
    m = _mgr()
    top = m.search_skills("reviewing and critiquing a draft", top_k=2)
    assert "review_checklist" in [s.name for s in top]


def test_metadata_for_filters_topk():
    """to_metadata_text_for 只列 top_k。"""
    m = _mgr()
    text = m.to_metadata_text_for("writing a literature survey", top_k=2)
    assert text.count("<skill") == 2


def test_search_all_when_fewer_than_topk():
    """skill 数 <= top_k → 全返回，不崩。"""
    m = _mgr()
    top = m.search_skills("anything", top_k=99)
    assert len(top) == len(m.list_names())


def test_metadata_for_degrades_to_full_on_error(monkeypatch):
    """语义检索失败 → 降级全列（不崩，全暴露）。"""
    m = _mgr()
    def _boom(*a, **k):
        raise RuntimeError("embedder OOM")
    monkeypatch.setattr(m, "search_skills", _boom)
    text = m.to_metadata_text_for("writing a survey", top_k=2)
    assert text.count("<skill") == len(m.list_names())   # 降级全列


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
    """skill_manager=None → 降级，不崩。"""
    tool = make_load_skill_tool(None)
    assert "not available" in await tool.ainvoke({"name": "cv"})
