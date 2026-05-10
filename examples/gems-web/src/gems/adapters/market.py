from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import override

from gems.app import GemMarketData
from gems.domain import GemMarketQuote, GemName
from inlay import Registry


@dataclass
class ReferenceGemMarketData(GemMarketData):
    @override
    async def quote_for[GemNameT: GemName](
        self,
        name: GemNameT,
    ) -> GemMarketQuote[GemNameT]:
        match name:
            case 'Emerald':
                return self._quote(name, Decimal('2200'), Decimal('0.18'))
            case 'Ruby':
                return self._quote(name, Decimal('3100'), Decimal('0.15'))
            case 'Sapphire':
                return self._quote(name, Decimal('1600'), Decimal('0.16'))
            case 'Diamond':
                return self._quote(name, Decimal('4500'), Decimal('0.12'))
            case 'Amethyst':
                return self._quote(name, Decimal('280'), Decimal('0.25'))

    def _quote[GemNameT: GemName](
        self,
        name: GemNameT,
        asking_price_per_carat_usd: Decimal,
        marketability_discount: Decimal,
    ) -> GemMarketQuote[GemNameT]:
        return GemMarketQuote(
            gem_name=name,
            asking_price_per_carat_usd=asking_price_per_carat_usd,
            marketability_discount=marketability_discount,
            source='reference-market-feed',
            quoted_at=datetime.now(UTC),
        )


MARKET_REGISTRY = Registry().register(GemMarketData)(ReferenceGemMarketData)
