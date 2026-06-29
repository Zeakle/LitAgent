"""Procedural Memory — PostgreSQL + FileSystem。

存储可复用的方法模板（Skills）。SQL 做匹配查询，YAML 文件做人类可读的版本化管理。
"""


import asyncpg

from litagent.config import MemoryConfig
from litagent.logging import get_logger

logger = get_logger('memory.procedural')


class ProceduralMemory:
    """Procedural Memory 存储层"""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    
    @staticmethod
    async def connect(config: MemoryConfig) -> "ProceduralMemory":
        pool = await asyncpg.create_pool(config.pg_url)
        logger.info("Connected to PostgreSQL (Procedural Memory)")
        return ProceduralMemory(pool)

    async def store(self, name: str, description: str, patterns: list[str], version: str = '1.0.0', source_file: str = '') -> str:
        """存储一个 Procedure 模板。返回 procedure_id。"""
        row = await self._pool.fetchrow(
            """INSERT INTO procedures (name, version, description, source_file)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (name, version) DO UPDATE SET description = $3
               RETURNING id""",
            name, version, description, source_file
        )
        proc_id = row['id']

        # 写入触发词
        for pattern in patterns:
            await self._pool.execute(
                """INSERT INTO trigger_patterns (procedure_id, pattern)
                   VALUES ($1, $2) ON CONFLICT DO NOTHING""",
                proc_id, pattern
            )
        
        logger.debug(f'Stored procedure: {name} v{version}')
        return str(proc_id)

    
    async def match(self, user_input: str) -> list[dict]:
        """关键词匹配 Procedure触发词"""
        rows = await self._pool.fetch(
            """SELECT p.*, array_agg(tp.pattern) as patterns
               FROM procedures p
               JOIN trigger_patterns tp ON tp.procedure_id = p.id
               WHERE p.status = 'active' AND $1 LIKE '%' || LOWER(tp.pattern) || '%'
               GROUP BY p.id
               LIMIT 5""",
            user_input.lower()
        )
        return [dict(r) for r in rows]

    
    async def record_execution(self, procedure_id: str, success: bool, duration_ms: int) -> None:
        """记录一次执行（更新计数器)"""
        field = "success_count" if success else "failure_count"
        await self._pool.execute(
            f"""UPDATE procedures SET {field} = {field} + 1,
                last_executed_at = now(), updated_at = now()
                WHERE id = $1""",
            procedure_id
        )


    async def close(self) -> None:
        """关闭PostgreSQL连接池"""
        await self._pool.close()