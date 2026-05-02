"""Shared fixtures for compile tests."""

import pytest

from inlay import RuleGraph
from inlay.default import default_rules


@pytest.fixture
def rules() -> RuleGraph:
    return default_rules()
