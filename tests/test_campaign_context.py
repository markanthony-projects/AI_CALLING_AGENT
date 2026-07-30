"""The price range the agent reads out on a live call.

min_price/max_price are stored in Lakhs while config_json quotes Crores. Emitting both
units left the model to reconcile them mid-call, and it told a prospect the project ran
"1 to 2 Crores" when the units are 1.2 to 3.5 Crores.
"""

import pytest

from app.utils.context_builder import _price_to_crores, build_campaign_context

SEEDED = {
    "name": "Lakeview Residency",
    "locality": "Whitefield",
    "min_price": 120.0,
    "max_price": 250.0,
    "config_json": [
        {"type": "2 BHK", "area": "1200 sqft", "price": "1.2 Cr"},
        {"type": "3 BHK", "area": "1600 sqft", "price": "1.8 Cr"},
        {"type": "Villament", "area": "2500 sqft", "price": "3.5 Cr"},
    ],
}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.2 Cr", 1.2),
        ("1.8 Crore", 1.8),
        ("3.5 Crores", 3.5),
        ("85 Lakh", 0.85),
        ("250 Lakhs", 2.5),
        (120.0, 1.2),  # bare numbers are the Lakhs columns
        (250, 2.5),
        (None, None),
        ("call for price", None),
    ],
)
def test_price_parsing(raw, expected):
    got = _price_to_crores(raw)
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_range_is_stated_in_crores_not_raw_lakhs():
    line = _range_line(build_campaign_context(SEEDED))
    assert "Lakhs" not in line, "raw Lakhs alongside Crore unit prices is what caused the misquote"
    assert "Crores" in line


def test_range_covers_every_unit_the_agent_can_quote():
    """The 3.5 Cr Villament sits outside max_price; the agent must not contradict itself."""
    line = _range_line(build_campaign_context(SEEDED))
    assert "1.2 Crores" in line
    assert "3.5 Crores" in line


def test_range_never_understates_the_floor():
    """It announced a 1 Crore floor on a project starting at 1.2 Crores."""
    line = _range_line(build_campaign_context(SEEDED))
    assert "1 Crores to" not in line
    assert "120" not in line and "250" not in line


def test_trailing_zeros_are_trimmed():
    ctx = build_campaign_context({"name": "P", "locality": "L", "min_price": 200.0, "max_price": 300.0})
    line = _range_line(ctx)
    assert "2 Crores to 3 Crores" in line


def test_missing_prices_omit_the_range():
    ctx = build_campaign_context({"name": "P", "locality": "L"})
    assert "Price Range" not in ctx


def test_unparseable_config_prices_fall_back_to_the_columns():
    ctx = build_campaign_context(
        {"name": "P", "locality": "L", "min_price": 120.0, "max_price": 250.0,
         "config_json": [{"type": "2 BHK", "price": "on request"}]}
    )
    assert "1.2 Crores to 2.5 Crores" in _range_line(ctx)


def _range_line(context: str) -> str:
    lines = [l for l in context.splitlines() if l.startswith("Overall Price Range")]
    return lines[0] if lines else ""
