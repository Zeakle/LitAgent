"""Prompt Injection 检测——规则扫描已知注入模式。"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from litagent.logging import get_logger


logger = get_logger('safety.injection')


class InjectionRisk(str, Enum):
    NONE = 'none'
    SUSPICIOUS = 'suspicious'  # 弱信号，隔离+警告
    HIGH = 'high'


@dataclass
class DetectionResult:
    risk: InjectionRisk
    matched: list[str]


# 高危模式：明确的越狱/指令覆盖意图
_HIGH_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
    r"disregard\s+(the\s+)?(above|previous|system)",
    r"you\s+are\s+now\s+(a|an)\b",
    r"forget\s+(everything|all|your)\s+(instructions?|rules?)",
    r"new\s+(system\s+)?(instructions?|prompt)\s*[:：]",
    r"do\s+not\s+(mention|reveal|tell)\s+this",
    r"<\s*system[_\-]?(prompt|reminder)\s*>",   # 伪造系统标签
]

# 可疑模式：弱信号，可能是正常学术内容，也可能是注入
_SUSPICIOUS_PATTERNS = [
    r"\bsystem\s+prompt\b",
    r"\bjailbreak\b",
    r"act\s+as\s+(a|an)\b",
    r"pretend\s+(you|to\s+be)\b",
]


class InjectionDetector:
    """规则式 prompt injection 检测器。"""

    def __init__(self):
        self._high = [re.compile(p, re.IGNORECASE) for p in _HIGH_PATTERNS]
        self._suspicious = [re.compile(p, re.IGNORECASE) for p in _SUSPICIOUS_PATTERNS]

    
    def scan(self, text: str) -> DetectionResult:
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