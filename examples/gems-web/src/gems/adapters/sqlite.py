from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, override

import aiosqlite

from gems.adapters.utils import gem_id_from_row, sqlite_gem_from_row
from gems.app import GemsRepository
from gems.domain import Gem, GemAppraisal, GemId, GemName, UserId
from inlay import Registry, qual

LIST_GEM_IDS_SQL = """
SELECT id
FROM gems
WHERE user_id = ?
ORDER BY created_at DESC
"""

GET_GEM_SQL = """
SELECT id,
       name,
       carat,
       appraisal_usd,
       appraisal_source,
       appraised_at,
       created_at,
       updated_at
FROM gems
WHERE id = ? AND user_id = ?
"""

CREATE_GEM_SQL = """
INSERT INTO gems (
    user_id,
    name,
    carat,
    appraisal_usd,
    appraisal_source,
    appraised_at,
    created_at,
    updated_at
)
VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?)
RETURNING id,
          name,
          carat,
          appraisal_usd,
          appraisal_source,
          appraised_at,
          created_at,
          updated_at
"""

UPDATE_GEM_SQL = """
UPDATE gems
SET name = ?,
    carat = ?,
    appraisal_usd = NULL,
    appraisal_source = NULL,
    appraised_at = NULL,
    updated_at = ?
WHERE id = ? AND user_id = ?
RETURNING id,
          name,
          carat,
          appraisal_usd,
          appraisal_source,
          appraised_at,
          created_at,
          updated_at
"""

DELETE_GEM_SQL = """
DELETE FROM gems
WHERE id = ? AND user_id = ?
"""

SAVE_APPRAISAL_SQL = """
UPDATE gems
SET appraisal_usd = ?,
    appraisal_source = ?,
    appraised_at = ?,
    updated_at = ?
WHERE id = ? AND user_id = ?
RETURNING id,
          name,
          carat,
          appraisal_usd,
          appraisal_source,
          appraised_at,
          created_at,
          updated_at
"""


@dataclass
class SQLiteGemsRepository(GemsRepository):
    user_id: UserId
    sqlite_path: Annotated[str, qual('sqlite_path')]

    @override
    async def list_gem_ids(self) -> list[GemId]:
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(LIST_GEM_IDS_SQL, (self.user_id,))
            async with cursor:
                return [gem_id_from_row(row) for row in await cursor.fetchall()]

    @override
    async def get_gem(self, gem_id: GemId) -> Gem | None:
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(GET_GEM_SQL, (gem_id, self.user_id))
            async with cursor:
                match await cursor.fetchone():
                    case None:
                        return None
                    case row:
                        return sqlite_gem_from_row(row)

    @override
    async def create_gem(self, name: GemName, carat: Decimal) -> Gem:
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                CREATE_GEM_SQL,
                (self.user_id, name, str(carat), now, now),
            )
            async with cursor:
                match await cursor.fetchone():
                    case None:
                        raise RuntimeError('Failed to create gem')
                    case row:
                        gem = sqlite_gem_from_row(row)
            await db.commit()
            return gem

    @override
    async def update_gem(
        self,
        gem_id: GemId,
        name: GemName,
        carat: Decimal,
    ) -> Gem | None:
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                UPDATE_GEM_SQL,
                (name, str(carat), datetime.now(UTC).isoformat(), gem_id, self.user_id),
            )
            async with cursor:
                match await cursor.fetchone():
                    case None:
                        return None
                    case row:
                        gem = sqlite_gem_from_row(row)
            await db.commit()
            return gem

    @override
    async def delete_gem(self, gem_id: GemId) -> bool:
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(DELETE_GEM_SQL, (gem_id, self.user_id))
            await db.commit()
            return cursor.rowcount > 0

    @override
    async def save_appraisal(
        self,
        gem_id: GemId,
        appraisal: GemAppraisal,
    ) -> Gem | None:
        appraised_at = appraisal.appraised_at.isoformat()
        async with aiosqlite.connect(self.sqlite_path) as db:
            cursor = await db.execute(
                SAVE_APPRAISAL_SQL,
                (
                    str(appraisal.amount_usd),
                    appraisal.source,
                    appraised_at,
                    appraised_at,
                    gem_id,
                    self.user_id,
                ),
            )
            async with cursor:
                match await cursor.fetchone():
                    case None:
                        return None
                    case row:
                        gem = sqlite_gem_from_row(row)
            await db.commit()
            return gem


SQLITE_REGISTRY = Registry().register(GemsRepository)(SQLiteGemsRepository)
