"""Phase 9 Skills module tests."""

import pytest
from pathlib import Path
from litagent.skills.manager import Skill, SkillManager, _find_skills_dir


# ── Skill dataclass ──

class TestSkill:
    def test_basic_skill(self):
        s = Skill(name="test", description="A test skill", body="# Body")
        assert s.name == "test"
        assert s.description == "A test skill"
        assert s.body == "# Body"

    def test_default_body_empty(self):
        s = Skill(name="x", description="x")
        assert s.body == ""


# ── SkillManager ──

class TestSkillManager:
    """Tests that SkillManager loads SKILL.md files correctly."""

    def test_loads_cv_and_nlp(self):
        m = SkillManager()
        names = m.list_names()
        assert "cv" in names
        assert "nlp" in names

    def test_to_metadata_text_contains_both_skills(self):
        m = SkillManager()
        text = m.to_metadata_text()
        assert 'name="cv"' in text
        assert 'name="nlp"' in text
        assert "<skill" in text
        assert "</skill>" in text

    def test_get_body_returns_markdown(self):
        m = SkillManager()
        body = m.get_body("cv")
        assert "# CV Paper Extraction" in body
        assert "model_architecture" in body
        assert "Quality checklist" in body

    def test_get_returns_skill_object(self):
        m = SkillManager()
        s = m.get("nlp")
        assert isinstance(s, Skill)
        assert s.name == "nlp"
        assert len(s.body) > 0

    def test_get_unknown_raises_keyerror(self):
        m = SkillManager()
        with pytest.raises(KeyError, match="unknown_skill"):
            m.get("unknown_skill")

    def test_skill_has_description(self):
        m = SkillManager()
        s = m.get("cv")
        assert "computer vision" in s.description.lower()

    def test_body_contains_quality_checklist(self):
        m = SkillManager()
        body = m.get_body("cv")
        assert "Quality checklist" in body.lower() or "quality checklist" in body.lower()

    def test_nlp_body_contains_fields(self):
        m = SkillManager()
        body = m.get_body("nlp")
        assert "model_type" in body
        assert "BLEU" in body


# ── _find_skills_dir ──

class TestFindSkillsDir:
    def test_finds_skills_dir(self):
        path = _find_skills_dir("src/litagent/skills")
        assert path.is_dir()
        assert (path / "cv" / "SKILL.md").is_file()   # cv 已扁平到 skills/ 直下（13.5 迁移）

    def test_raises_on_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            _find_skills_dir("nonexistent_dir_xyz")
