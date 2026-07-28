"""Load local skills and select relevant metadata for agent prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from litagent.logging import get_logger
from litagent.context.templates import wrap_xml
from litagent.rag.embedder import get_embedder

logger = get_logger("skills.manager")


@dataclass
class Skill:
    """Represent parsed skill metadata, content, and cached embedding."""

    name: str
    description: str
    body: str = ""
    embedding: list[float] | None = None
    path: Path = field(default_factory=Path)


class SkillManager:
    """Discover local skills and expose semantic metadata retrieval."""

    def __init__(self, skills_dir: str = "src/litagent/skills"):
        self._skills: dict[str, Skill] = {}
        skill_path = _find_skills_dir(skills_dir)
        logger.debug(f"Scanning skills from: {skill_path}")
        self._scan(skill_path)
        logger.info(f"Loaded {len(self._skills)} skills: {self.list_names()}")

    def _scan(self, skill_path: Path) -> None:
        """Load every valid immediate child skill directory."""
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

    @staticmethod
    def _parse_frontmatter(md_file: Path) -> Skill:
        """Parse required scalar metadata and body from one SKILL.md."""
        text = md_file.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError(
                f"{md_file}: Missing YAML frontmatter (must start with ---)"
            )

        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(
                f"{md_file}: Unclosed YAML frontmatter (missing closing ---)"
            )

        frontmatter_text = parts[1].strip()
        body = parts[2].strip()

        # Skill metadata uses flat scalar keys, so full YAML parsing is unnecessary.
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
            raise ValueError(f"{md_file}: 'description' is required in frontmatter")

        if len(name) > 64:
            raise ValueError(f"{md_file}: name exceeds 64 characters ({len(name)})")

        return Skill(name=name, description=description, body=body, path=md_file)

    def _ensure_embeddings(self) -> None:
        """Populate missing description embeddings in place."""
        missing = [s for s in self._skills.values() if s.embedding is None]

        if not missing:
            return
        # Cache embeddings on each skill for subsequent semantic searches.
        embedder = get_embedder()
        vecs = embedder.embed([s.description for s in missing])
        for s, v in zip(missing, vecs):
            s.embedding = v

    def search_skills(self, query: str, top_k: int = 3) -> list[Skill]:
        """Return the skills most semantically relevant to a query."""
        skills = list(self._skills.values())
        if len(skills) <= top_k:
            return skills

        self._ensure_embeddings()
        embedder = get_embedder()
        qv = embedder.embed(query)
        scored = [
            (sum(a * b for a, b in zip(qv, s.embedding)), s)
            for s in skills
            if s.embedding
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:top_k]]

    def to_metadata_text_for(self, query: str, top_k: int = 3) -> str:
        """Render relevant skill descriptions for prompt metadata."""
        try:
            skills = self.search_skills(query, top_k)
        except Exception as e:
            logger.warning(
                f"Semantic skill search failed ({e}), falling back to full list"
            )
            return self.to_metadata_text()
        if not skills:
            return ""
        parts = ["## Available Skills", ""]
        for s in skills:
            parts.append(wrap_xml("skill", s.description, {"name": s.name}))
            parts.append("")

        return "\n".join(parts)

    def to_metadata_text(self) -> str:
        """Render every skill description for prompt metadata."""
        if not self._skills:
            return ""
        parts = ["## Available Skills", ""]
        for s in self._skills.values():
            parts.append(wrap_xml("skill", s.description, {"name": s.name}))
            parts.append("")
        return "\n".join(parts)

    def get_body(self, name: str) -> str:
        """Return the full body of a named skill."""
        skill = self.get(name)
        return skill.body

    def get(self, name: str) -> Skill:
        """Return a named skill or raise an informative KeyError."""
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not found. Available: {self.list_names()}")
        return self._skills[name]

    def list_names(self) -> list[str]:
        """Return loaded skill names in discovery order."""
        return list(self._skills.keys())


def _find_skills_dir(skills_dir: str) -> Path:
    """Resolve an explicit or package-relative skills directory."""
    path = Path(skills_dir)
    if path.is_dir():
        return path

    root = Path(__file__).resolve().parent
    path = root / skills_dir
    if path.is_dir():
        return path
    raise FileNotFoundError(
        f"Skills directory not found: {skills_dir} " f"(also tried {path})"
    )
