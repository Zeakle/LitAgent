"""Build XML-delimited prompt sections."""

from __future__ import annotations


def wrap_xml(tag: str, content: str, attrs: dict[str, str] | None = None) -> str:
    """Wrap non-empty content in an XML-like element."""
    if not content.strip():
        return ""
    if attrs:
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        opening = f"<{tag} {attr_str}>"
    else:
        opening = f"<{tag}>"
    return f"{opening}\n{content}\n</{tag}>"


def build_system_prompt(
    role: str,
    instructions: str,
    tools: str = "",
    skills: str = "",
    context: str = "",
    constraints: str = "",
) -> str:
    """Assemble a system prompt from non-empty XML-delimited sections."""
    parts: list[str] = [
        wrap_xml("role", role),
        wrap_xml("instructions", instructions),
    ]

    if tools:
        parts.append(wrap_xml("available_tools", tools))
    if skills:
        parts.append(wrap_xml("available_skills", skills))
    if context:
        parts.append(wrap_xml("context", context))
    if constraints:
        parts.append(wrap_xml("constraints", constraints))

    return "\n\n".join(p for p in parts if p)
