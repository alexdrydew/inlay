import datetime
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, NewType

UserId = NewType('UserId', str)
GemId = NewType('GemId', int)

GemName = Literal['Emerald', 'Ruby', 'Sapphire', 'Diamond', 'Amethyst']


@dataclass
class Gem[GemNameT: GemName = GemName]:
    id: GemId
    name: GemNameT
    carat: Decimal
    appraisal_usd: Decimal | None
    appraisal_source: str | None
    appraised_at: datetime.datetime | None
    created_at: datetime.datetime
    updated_at: datetime.datetime


@dataclass(frozen=True)
class GemMarketQuote[GemNameT: GemName = GemName]:
    gem_name: GemNameT
    asking_price_per_carat_usd: Decimal
    marketability_discount: Decimal
    source: str
    quoted_at: datetime.datetime


@dataclass(frozen=True)
class GemAppraisal:
    amount_usd: Decimal
    source: str
    appraised_at: datetime.datetime


def gem_name(name: str) -> GemName:
    match name:
        case 'Emerald' | 'Ruby' | 'Sapphire' | 'Diamond' | 'Amethyst':
            return name
        case _:
            raise ValueError(f'Invalid gem name: {name}')


def appraise_gem[GemNameT: GemName](
    gem: Gem[GemNameT],
    quote: GemMarketQuote[GemNameT],
    *,
    appraised_at: datetime.datetime,
) -> GemAppraisal:
    if quote.asking_price_per_carat_usd <= 0:
        raise ValueError('Market quote price must be positive')
    if not Decimal('0') <= quote.marketability_discount < Decimal('1'):
        raise ValueError('Marketability discount must be in [0, 1)')

    if gem.carat <= 0:
        raise ValueError('Gem carat must be positive')

    amount = (
        quote.asking_price_per_carat_usd
        * gem.carat
        * (Decimal('1') - quote.marketability_discount)
    )
    return GemAppraisal(
        amount_usd=amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
        source=quote.source,
        appraised_at=appraised_at,
    )
