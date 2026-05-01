from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from gems.domain import Gem, GemId, GemName


def postgres_gem_from_row(row: Sequence[object]) -> Gem:
    match row:
        case (
            int(id_),
            str(name),
            float() | Decimal() as carat,
            datetime() as created_at,
            datetime() as updated_at,
        ):
            return Gem(
                id=GemId(id_),
                name=gem_name(name),
                carat=float(carat),
                created_at=created_at,
                updated_at=updated_at,
            )
        case _:
            raise ValueError('Invalid gem row')


def sqlite_gem_from_row(row: Sequence[object]) -> Gem:
    match row:
        case (
            int(id_),
            str(name),
            int() | float() as carat,
            str(created_at),
            str(updated_at),
        ):
            return Gem(
                id=GemId(id_),
                name=gem_name(name),
                carat=float(carat),
                created_at=datetime.fromisoformat(created_at),
                updated_at=datetime.fromisoformat(updated_at),
            )
        case _:
            raise ValueError('Invalid gem row')


def gem_name(name: str) -> GemName:
    match name:
        case 'Emerald' | 'Ruby' | 'Sapphire' | 'Diamond' | 'Amethyst':
            return name
        case _:
            raise ValueError(f'Invalid gem name: {name}')
