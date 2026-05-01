from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from gems.domain import Gem, GemId, gem_name


def gem_id_from_row(row: Sequence[object]) -> GemId:
    match row:
        case (int(id_),):
            return GemId(id_)
        case _:
            raise ValueError('Invalid gem id row')


def decimal_from_value(value: object) -> Decimal:
    match value:
        case Decimal() as amount:
            return amount
        case int() | float() | str() as amount:
            return Decimal(str(amount))
        case _:
            raise ValueError('Invalid decimal value')


def decimal_or_none(value: object) -> Decimal | None:
    match value:
        case None:
            return None
        case _:
            return decimal_from_value(value)


def datetime_or_none(value: object) -> datetime | None:
    match value:
        case None:
            return None
        case datetime() as timestamp:
            return timestamp
        case str() as timestamp:
            return datetime.fromisoformat(timestamp)
        case _:
            raise ValueError('Invalid datetime value')


def str_or_none(value: object) -> str | None:
    match value:
        case None:
            return None
        case str() as text:
            return text
        case _:
            raise ValueError('Invalid string value')


def postgres_gem_from_row(row: Sequence[object]) -> Gem:
    match row:
        case (
            int(id_),
            str(name),
            int() | float() | Decimal() | str() as carat,
            appraisal_usd,
            appraisal_source,
            appraised_at,
            datetime() as created_at,
            datetime() as updated_at,
        ):
            return Gem(
                id=GemId(id_),
                name=gem_name(name),
                carat=decimal_from_value(carat),
                appraisal_usd=decimal_or_none(appraisal_usd),
                appraisal_source=str_or_none(appraisal_source),
                appraised_at=datetime_or_none(appraised_at),
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
            int() | float() | str() as carat,
            appraisal_usd,
            appraisal_source,
            appraised_at,
            str(created_at),
            str(updated_at),
        ):
            return Gem(
                id=GemId(id_),
                name=gem_name(name),
                carat=decimal_from_value(carat),
                appraisal_usd=decimal_or_none(appraisal_usd),
                appraisal_source=str_or_none(appraisal_source),
                appraised_at=datetime_or_none(appraised_at),
                created_at=datetime.fromisoformat(created_at),
                updated_at=datetime.fromisoformat(updated_at),
            )
        case _:
            raise ValueError('Invalid gem row')
