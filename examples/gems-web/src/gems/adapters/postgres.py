from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, override

from psycopg import AsyncConnection

from gems.adapters.utils import gem_id_from_row, postgres_gem_from_row
from gems.app import GemsRepository
from gems.domain import Gem, GemAppraisal, GemId, GemName, UserId
from inlay import Registry, qual

LIST_GEM_IDS_SQL = """
SELECT id
FROM gems
WHERE user_id = %s
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
WHERE id = %s AND user_id = %s
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
VALUES (%s, %s, %s, NULL, NULL, NULL, %s, %s)
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
SET name = %s,
    carat = %s,
    appraisal_usd = NULL,
    appraisal_source = NULL,
    appraised_at = NULL,
    updated_at = %s
WHERE id = %s AND user_id = %s
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
WHERE id = %s AND user_id = %s
"""

SAVE_APPRAISAL_SQL = """
UPDATE gems
SET appraisal_usd = %s,
    appraisal_source = %s,
    appraised_at = %s,
    updated_at = %s
WHERE id = %s AND user_id = %s
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
class PostgresGemsRepository(GemsRepository):
    user_id: UserId
    psql_uri: Annotated[str, qual('psql_uri')]

    @override
    async def list_gem_ids(self) -> list[GemId]:
        async with await AsyncConnection.connect(self.psql_uri) as conn:
            async with conn.cursor() as cursor:
                _ = await cursor.execute(LIST_GEM_IDS_SQL, (self.user_id,))
                return [gem_id_from_row(row) for row in await cursor.fetchall()]

    @override
    async def get_gem(self, gem_id: GemId) -> Gem | None:
        async with await AsyncConnection.connect(self.psql_uri) as conn:
            async with conn.cursor() as cursor:
                _ = await cursor.execute(GET_GEM_SQL, (gem_id, self.user_id))
                match await cursor.fetchone():
                    case None:
                        return None
                    case row:
                        return postgres_gem_from_row(row)

    @override
    async def create_gem(self, name: GemName, carat: Decimal) -> Gem:
        now = datetime.now(UTC)
        async with await AsyncConnection.connect(self.psql_uri) as conn:
            async with conn.cursor() as cursor:
                _ = await cursor.execute(
                    CREATE_GEM_SQL,
                    (self.user_id, name, carat, now, now),
                )
                match await cursor.fetchone():
                    case None:
                        raise RuntimeError('Failed to create gem')
                    case row:
                        return postgres_gem_from_row(row)

    @override
    async def update_gem(
        self,
        gem_id: GemId,
        name: GemName,
        carat: Decimal,
    ) -> Gem | None:
        async with await AsyncConnection.connect(self.psql_uri) as conn:
            async with conn.cursor() as cursor:
                _ = await cursor.execute(
                    UPDATE_GEM_SQL,
                    (name, carat, datetime.now(UTC), gem_id, self.user_id),
                )
                match await cursor.fetchone():
                    case None:
                        return None
                    case row:
                        return postgres_gem_from_row(row)

    @override
    async def delete_gem(self, gem_id: GemId) -> bool:
        async with await AsyncConnection.connect(self.psql_uri) as conn:
            async with conn.cursor() as cursor:
                _ = await cursor.execute(DELETE_GEM_SQL, (gem_id, self.user_id))
                return cursor.rowcount > 0

    @override
    async def save_appraisal(
        self,
        gem_id: GemId,
        appraisal: GemAppraisal,
    ) -> Gem | None:
        async with await AsyncConnection.connect(self.psql_uri) as conn:
            async with conn.cursor() as cursor:
                _ = await cursor.execute(
                    SAVE_APPRAISAL_SQL,
                    (
                        appraisal.amount_usd,
                        appraisal.source,
                        appraisal.appraised_at,
                        appraisal.appraised_at,
                        gem_id,
                        self.user_id,
                    ),
                )
                match await cursor.fetchone():
                    case None:
                        return None
                    case row:
                        return postgres_gem_from_row(row)


POSTGRES_REGISTRY = Registry().register(GemsRepository)(PostgresGemsRepository)
