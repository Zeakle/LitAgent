"""Detect common prompt-injection indicators in untrusted text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from litagent.logging import get_logger

logger = get_logger("safety.injection")


class InjectionRisk(str, Enum):
    """Classify prompt injection as absent, suspicious, or high risk."""

    NONE = "none"
    SUSPICIOUS = "suspicious"
    HIGH = "high"


@dataclass
class DetectionResult:
    """Represent a prompt-injection risk classification and its matches."""

    risk: InjectionRisk
    matched: list[str]


_HIGH_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
    r"disregard\s+(the\s+)?(above|previous|system)",
    r"you\s+are\s+now\s+(a|an)\b",
    r"forget\s+(everything|all|your)\s+(instructions?|rules?)",
    r"new\s+(system\s+)?(instructions?|prompt)\s*[:：]",
    r"do\s+not\s+(mention|reveal|tell)\s+this",
    r"<\s*system[_\-]?(prompt|reminder)\s*>",
]


_SUSPICIOUS_PATTERNS = [
    r"\bsystem\s+prompt\b",
    r"\bjailbreak\b",
    r"act\s+as\s+(a|an)\b",
    r"pretend\s+(you|to\s+be)\b",
]


class InjectionDetector:
    """Classify text with compiled high-risk and suspicious patterns."""

    def __init__(self):
        """Initialize the injection detector."""
        self._high = [re.compile(p, re.IGNORECASE) for p in _HIGH_PATTERNS]
        self._suspicious = [re.compile(p, re.IGNORECASE) for p in _SUSPICIOUS_PATTERNS]

    def scan(self, text: str) -> DetectionResult:
        """Scan text for prompt-injection indicators."""
        if not text:
            return DetectionResult(InjectionRisk.NONE, [])

        high_hits = [p.pattern for p in self._high if p.search(text)]
        if high_hits:
            logger.warning(f"HIGH injection risk: {high_hits}")
            return DetectionResult(InjectionRisk.HIGH, high_hits)

        susp_hits = [p.pattern for p in self._suspicious if p.search(text)]
        if susp_hits:
            logger.info(f"Suspicious patterns: {susp_hits}")
            return DetectionResult(InjectionRisk.SUSPICIOUS, susp_hits)

        return DetectionResult(InjectionRisk.NONE, [])
