"""Provide deterministic regex-based extraction tools."""

import re

from litagent.tools.base import ToolCategory, ToolDefinition
from litagent.tools.registry import ToolRegistry, get_registry


async def extract_claims(abstract: str = "") -> list[str]:
    """Extract up to five claim-like sentences from an abstract."""
    if not abstract:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    keywords = [
        "achieve",
        "outperform",
        "surpass",
        "state-of-the-art",
        "sota",
        "best",
        "novel",
        "first",
        "superior",
    ]
    return [s.strip() for s in sentences if any(kw in s.lower() for kw in keywords)][:5]


async def extract_metrics(abstract: str = "") -> dict[str, str]:
    """Extract percentage metrics keyed by their preceding labels."""
    metrics = {}
    for m in re.finditer(r"(\w+)\s*(?:of|=|:)\s*(\d+\.?\d*)\s*%", abstract):
        metrics[m.group(1).lower()] = f"{m.group(2)}%"
    return metrics


async def extract_methods(text: str = "") -> list[str]:
    """Extract up to three proposed method names from text."""
    methods = []
    for m in re.finditer(
        r"(?:propose|introduce|present)\s+(?:a\s+)?(\w+(?:\s+\w+){0,2})", text
    ):
        methods.append(m.group(1).strip())
    return methods[:3]


async def extract_datasets(text: str = "") -> list[str]:
    """Return known dataset names mentioned in text."""
    known = [
        "imagenet",
        "miniImageNet",
        "tieredImageNet",
        "cifar",
        "coco",
        "voc",
        "meta-dataset",
        "omniglot",
    ]
    return [d for d in known if d.lower() in text]


def register_extract_tools(registry: ToolRegistry | None = None) -> None:
    """Register extraction tools in an injected or compatibility registry."""
    r = registry if registry is not None else get_registry()
    r.register(
        ToolDefinition(
            name="extract_claims",
            description="Extract claims from abstract",
            parameters={
                "type": "object",
                "properties": {"abstract": {"type": "string"}},
                "required": ["abstract"],
                "additionalProperties": False,
            },
            category=ToolCategory.READ,
        ),
        extract_claims,
    )
    r.register(
        ToolDefinition(
            name="extract_metrics",
            description="Extract numerical metrics",
            parameters={
                "type": "object",
                "properties": {"abstract": {"type": "string"}},
                "required": ["abstract"],
                "additionalProperties": False,
            },
            category=ToolCategory.READ,
        ),
        extract_metrics,
    )
    r.register(
        ToolDefinition(
            name="extract_methods",
            description="Extract method names",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            category=ToolCategory.READ,
        ),
        extract_methods,
    )
    r.register(
        ToolDefinition(
            name="extract_datasets",
            description="Extract dataset names",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            category=ToolCategory.READ,
        ),
        extract_datasets,
    )
