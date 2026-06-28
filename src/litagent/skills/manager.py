"""Skill Manager — 对接 Anthropic Agent Skills 标准。

Skills 目录结构:
    skills_dir/
    ├── cv/
    │   └── SKILL.md       ← YAML frontmatter + Markdown body
    └── nlp/
        └── SKILL.md

每个 SKILL.md 格式:
    ---
    name: skill-name
    description: When to use this skill...
    ---
    # Markdown body (LLM 触发后注入的完整指令)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from litagent.logging import get_logger
from litagent.context.templates import wrap_xml

logger = get_logger("skills.manager")


# ═══════════════════════════════════════════════════════════
# Skill: 对应一个 SKILL.md 文件
#
# 对齐 Anthropic Agent Skills 标准的渐进式披露:
#   Level 1: name + description → 始终在 system prompt（~100 tokens）
#   Level 2: body → LLM 决定触发后注入完整 markdown 指令
#   Level 3: 目录下的 scripts/references/assets → 按需加载（Phase 9 暂不实现）
#
# name 和 description 来自 YAML frontmatter。
# body 是 --- 之后的全部 markdown 内容。
#
@dataclass
class Skill:
    """从 SKILL.md 加载的技能。"""
    name: str
    description: str
    body: str = ""
    path: Path = field(default_factory=Path)


# ═══════════════════════════════════════════════════════════
# SkillManager: 扫描目录 → 解析 SKILL.md → 渐进式披露
#
class SkillManager:
    """管理所有 Skill，提供渐进式披露。

    Level 1 — to_metadata_text():
      所有 skill 的 name + description。
      始终注入 system prompt，LLM 据此决定触发哪个 skill。

    Level 2 — get_body(name) / get(name):
      LLM 选中 skill 后，获取完整 markdown body 注入 context。

    用法:
        manager = SkillManager("src/litagent/skills/extraction")
        # Level 1: 注入所有 skill 的元数据
        metadata = manager.to_metadata_text()
        # Level 2: LLM 选了 cv → 拿到完整 body
        body = manager.get_body("cv")
    """

    def __init__(self, skills_dir: str = "src/litagent/skills/extraction"):
        self._skills: dict[str, Skill] = {}
        skill_path = _find_skills_dir(skills_dir)
        logger.debug(f"Scanning skills from: {skill_path}")
        self._scan(skill_path)
        logger.info(f"Loaded {len(self._skills)} skills: {self.list_names()}")

    # ── 扫描目录 ──

    def _scan(self, skill_path: Path) -> None:
        """递归扫描 skill_path，找到所有 */SKILL.md 并解析。

        和 Anthropic Claude Code 行为一致:
        - 扫描目录下的每个子目录
        - 子目录名 = skill 的目录名（不是 name 字段）
        - 每个子目录必须有 SKILL.md
        """
        for sk_dir in sorted(skill_path.iterdir()):
            if not sk_dir.is_dir():
                continue
            md_file = sk_dir / "SKILL.md"
            if not md_file.is_file():
                logger.debug(f"Skipping {sk_dir.name}: no SKILL.md")
                continue
            try:
                skill = self._parse_frontmatter(md_file)
                self._skills[skill.name] = skill
                logger.debug(f"  Loaded: [{skill.name}] from {sk_dir.name}/")
            except Exception as e:
                logger.warning(f"Failed to parse {md_file}: {e}")

        if not self._skills:
            raise FileNotFoundError(
                f"No SKILL.md files found under {skill_path}. "
                f"Expected structure: skills_dir/<name>/SKILL.md"
            )

    # ── 解析 YAML frontmatter ──

    @staticmethod
    def _parse_frontmatter(md_file: Path) -> Skill:
        """解析 SKILL.md 的 YAML frontmatter + Markdown body。

        SKILL.md 结构:
            ---
            name: skill-name
            description: When to use...
            ---
            # Markdown body

        用 --- 做分隔符。第一个 --- 必须在文件开头。
        """
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError(
                f"{md_file}: Missing YAML frontmatter (must start with ---)"
            )

        # 找到第二个 ---
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(
                f"{md_file}: Unclosed YAML frontmatter (missing closing ---)"
            )

        frontmatter_text = parts[1].strip()
        body = parts[2].strip()

        # 简单的 YAML frontmatter 解析（不引入 PyYAML 依赖）
        # SKILL.md 的 frontmatter 非常简单: 只有 name 和 description
        frontmatter: dict[str, str] = {}
        for line in frontmatter_text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                frontmatter[key.strip()] = value.strip()

        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")

        if not name:
            raise ValueError(f"{md_file}: 'name' is required in frontmatter")
        if not description:
            raise ValueError(
                f"{md_file}: 'description' is required in frontmatter"
            )

        # Anthropic 规范的 name 最长 64 字符
        if len(name) > 64:
            raise ValueError(
                f"{md_file}: name exceeds 64 characters ({len(name)})"
            )

        return Skill(name=name, description=description, body=body, path=md_file)

    # ── Level 1: 元数据（始终注入 system prompt）──

    def to_metadata_text(self) -> str:
        """所有 skill 的 name + description。

        对齐 Anthropic Agent Skills 的 Level 1 渐进式披露。
        注入 Extractor system prompt，LLM 根据 description 选择 skill。

        输出格式:
          <skill name="cv">
          Extract model architecture, backbone, datasets...
          </skill>
        """
        if not self._skills:
            return ""
        parts = ["## Available Skills", ""]
        for s in self._skills.values():
            parts.append(wrap_xml("skill", s.description, {"name": s.name}))
            parts.append("")
        return "\n".join(parts)

    # ── Level 2: 完整 body（LLM 触发后注入）──

    def get_body(self, name: str) -> str:
        """获取指定 skill 的完整 markdown body。

        LLM 根据 to_metadata_text() 选中 skill name 后，
        调用此方法拿到 Level 2 内容注入 context。
        """
        skill = self.get(name)
        return skill.body

    # ── 辅助 API ──

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(
                f"Skill '{name}' not found. Available: {self.list_names()}"
            )
        return self._skills[name]

    def list_names(self) -> list[str]:
        return list(self._skills.keys())


# ═══════════════════════════════════════════════════════════
# 辅助: 定位 skills 目录
#
def _find_skills_dir(skills_dir: str) -> Path:
    """定位 skills 目录。先尝试 cwd 相对路径，再从当前文件推算。"""
    path = Path(skills_dir)
    if path.is_dir():
        return path
    # 从当前文件推算: manager.py → skills/ → extraction/
    # 如果 skills_dir 是相对路径如 "skills/extraction"，相对于 src/litagent/ 找
    root = Path(__file__).resolve().parent
    path = root / skills_dir
    if path.is_dir():
        return path
    raise FileNotFoundError(
        f"Skills directory not found: {skills_dir} "
        f"(also tried {path})"
    )
