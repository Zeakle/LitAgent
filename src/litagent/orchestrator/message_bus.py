"""MessageBus——Agent 间异步通信。"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from litagent.logging import get_logger


logger = get_logger('orchestrator.message_bus')


class MessageType(str, Enum):
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = 'task_result'
    TASK_FAILED = 'task_failed'
    STATUS_QUERY = 'status_query'
    STATUS_REPORT = 'status_report'
    REPLAN_REQUEST = 'replan_request'
    SHARED_DISCOVERY = 'shared_discovery'


@dataclass
class AgentMessage:
    """Agent 间通信的消息结构"""
    type: MessageType
    sender: str
    receiver: str
    task_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """基于 asyncio.Queue的消息总线。
    
    每个注册的Agent的独立的收件箱(Queue).
    支持点对点send和广播broadcast

    Phase 8a 的 Scheduler 直接调 Worker.execute()，不经过 MessageBus。
    MessageBus 主要用于：
    1. Phase 8c 的 Synthesis↔Reviewer 对抗循环中的消息交换
    2. Worker 之间的 shared_discovery（跨 Worker 共享发现）
    3. 动态重规划请求（replan_request）
    """

    def __init__(self):
        # asyncio.Queue 异步安全队列，进队出队异步运行，队空时取元素操作自动等待
        self._queues: dict[str, asyncio.Queue[AgentMessage]] = {}

    
    def register(self, agent_id: str) -> None:
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()


    def unregister(self, agent_id: str) -> None:
        self._queues.pop(agent_id, None)

    
    async def send(self, msg: AgentMessage) -> None:
        """点对点发送, receiver不存在则丢弃(不崩溃)"""
        q = self._queues.get(msg.receiver)
        if q:
            await q.put(msg)
        else:
            logger.debug(f"Message to '{msg.receiver}' dropped: not registered")

    
    async def receive(self, agent_id: str, timeout: float = 30.0) -> AgentMessage | None:
        """从收件箱取一条消息，超时返回None"""
        q = self._queues.get(agent_id)
        if not q:
            return None
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


    async def broadcast(self, msg: AgentMessage, exclude_sender: bool = True) -> None:
        """广播给所有注册的Agent"""
        for aid, q in self._queues.items():
            if exclude_sender and aid == msg.sender:
                continue
            await q.put(msg)

    
    def pending_count(self, agent_id: str) -> int:
        q = self._queues.get(agent_id)
        return q.qsize() if q else 0