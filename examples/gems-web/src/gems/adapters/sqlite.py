from dataclasses import dataclass
from typing import Annotated, override

import aiosqlite

from gems.adapters.utils import sqlite_gem_from_row
from gems.app import GemsRepository
from gems.domain import Gem, UserId
from inlay import RegistryBuilder, qual

LIST_GEMS_SQL = """
SELECT id, name, carat, created_at, updated_at
FROM gems
WHERE user_id = ?
ORDER BY created_at DESC
"""


@dataclass
class SQLiteGemsRepository(GemsRepository):
    user_id: UserId
    sqlite_path: Annotated[str, qual('sqlite_path')]

    @override
    async def list_gems(self) -> list[Gem]:
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(LIST_GEMS_SQL, (self.user_id,))
            async with cursor:
                return [sqlite_gem_from_row(row) for row in await cursor.fetchall()]


SQLITE_REGISTRY = RegistryBuilder().register(GemsRepository)(SQLiteGemsRepository)
