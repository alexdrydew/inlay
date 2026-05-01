import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, NewType

UserId = NewType('UserId', str)
GemId = NewType('GemId', int)

GemName = Literal['Emerald', 'Ruby', 'Sapphire', 'Diamond', 'Amethyst']


@dataclass
class Gem:
    id: GemId
    name: GemName
    carat: float
    created_at: datetime.datetime
    updated_at: datetime.datetime


def appraise_gem_usd(_gem: Gem) -> Decimal: ...
