from dataclasses import dataclass
from typing import Annotated, override

from psycopg import AsyncConnection

from gems.adapters.utils import postgres_gem_from_row
from gems.app import GemsRepository
from gems.domain import Gem, UserId
from inlay import RegistryBuilder, qual

LIST_GEMS_SQL = """
SELECT id, name, carat, created_at, updated_at
FROM gems
WHERE user_id = %s
ORDER BY created_at DESC
"""


@dataclass
class PostgresGemsRepository(GemsRepository):
    user_id: UserId
    psql_uri: Annotated[str, qual('psql_uri')]

    @override
    async def list_gems(self) -> list[Gem]:
        async with await AsyncConnection.connect(self.psql_uri) as conn:
            async with conn.cursor() as cursor:
                _ = await cursor.execute(LIST_GEMS_SQL, (self.user_id,))
                return [postgres_gem_from_row(row) for row in await cursor.fetchall()]


POSTGRES_REGISTRY = RegistryBuilder().register(GemsRepository)(PostgresGemsRepository)
