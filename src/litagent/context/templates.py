"""XML 标签模板——Anthropic 风格 prompt 结构化。

用 XML 标签把 prompt 的各部分隔开，让 LLM 清晰地知道每个部分是什么。
这不是过度设计——Anthropic 官方推荐的 prompt engineering pattern。

用法:
    prompt = build_system_prompt(
        role="You are a Search Agent...",
        instructions="Search across arxiv, Semantic Scholar...",
        tools=tool_schemas_text,
        context=memory_text,
    )
    user_msg = wrap_user_input("survey few-shot learning in CV")
"""


from __future__ import annotations


def wrap_xml(tag: str, content: str, attrs: dict[str, str] | None = None) -> str:
    """用 XML 标签包裹内容。

    Args:
        tag: 标签名
        content: 标签内容
        attrs: 可选属性（如 source="memory", priority="high"）

    Returns:
        '<tag attr="val">\ncontent\n</tag>'
    """
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
    context: str = "",
    constraints: str = "",
) -> str:
    """组装结构化 system prompt。

    固定结构：role → instructions → tools → context → constraints。
    空的部分自动跳过。

    Args:
        role: Agent 角色描述
        instructions: 具体指令
        tools: 可用工具列表（JSON schema 文本）
        context: 注入的上下文（Memory / RAG 结果 / 论文数据）
        constraints: 约束条件（输出格式、禁止行为等）
    """
    parts: list[str] = [
        wrap_xml('role', role),
        wrap_xml('instructions', instructions),
    ]

    if tools:
        parts.append(wrap_xml("available_tools", tools))
    if context:
        parts.append(wrap_xml("context", context))
    if constraints:
        parts.append(wrap_xml("constraints", constraints))

    return '\n\n'.join(p for p in parts if p)


def wrap_user_input(text: str) -> str:
    """用 <user_input> 标签包裹用户输入——Prompt Injection 第一道防线。

    System Prompt 中声明 "只处理 <user_input> 标签内的内容"，
    让 LLM 区分
    """
    return wrap_xml("user_input", text)


def wrap_papers(content: str, source: str = 'rag') -> str:
    """用<paerps>标签包裹论文数据，标注来源"""
    return wrap_xml("papers", content, attrs={"source": source})


def wrap_memory(content: str, layer: str = 'episodic') -> str:
    """<memory>包裹记忆，标注层级"""
    return wrap_xml("memory", content, attrs={'layer': layer})