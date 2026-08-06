"""The agent must name the configurations the project actually sells.

From call 5664ace6, on Abhee Codename New Dimension:

    AGENT → "We have 2, 3, and 4 B H K homes starting from 1.17 Crores."

The project sells 2 BHK, 3 BHK (Regular, Comfort and Luxury), 3.5 BHK Presidential and
4.5 BHK Presidential. There is no 4 BHK. The 4.5 was rounded down and the 3.5 vanished, so
the agent offered a flat that does not exist and hid one that does.
"""

import pytest

from app.prompts.agent_prompts import get_system_prompt
from app.utils.context_builder import build_campaign_context, spoken_configurations

# Exactly as it sits in the projects table.
NEW_DIMENSION = [
    {"type": "2 BHK", "area": "1181 - 1183 sqft", "price": "1.17 Cr"},
    {"type": "3 BHK Regular", "area": "1448 - 1454 sqft", "price": "1.46 Cr"},
    {"type": "3 BHK Comfort", "area": "1550 - 1558 sqft", "price": "1.56 Cr"},
    {"type": "3 BHK Luxury", "area": "1646 - 1703 sqft", "price": "1.64 - 1.74 Cr"},
    {"type": "3.5 BHK Presidential", "area": "2009 sqft", "price": "2.06 Cr"},
    {"type": "4.5 BHK Presidential", "area": "2556 sqft", "price": "2.64 Cr"},
]


def test_the_call_that_went_wrong():
    assert spoken_configurations(NEW_DIMENSION) == [
        "2 B H K",
        "3 B H K",
        "3.5 B H K",
        "4.5 B H K",
    ]


def test_a_half_configuration_is_never_rounded():
    """The specific failure. 4.5 became 4, which is a flat the project does not sell."""
    units = spoken_configurations([{"type": "4.5 BHK Presidential"}])
    assert units == ["4.5 B H K"]
    assert "4 B H K" not in units


def test_nothing_the_project_sells_is_dropped():
    """3.5 disappeared entirely from the spoken list."""
    counts = {u.split(" B")[0] for u in spoken_configurations(NEW_DIMENSION)}
    assert counts == {"2", "3", "3.5", "4.5"}


def test_trim_levels_collapse_to_one_configuration():
    """Regular, Comfort and Luxury are one thing to name on an opening call. Reading all
    three back is a brochure, and the full table is still in the context when they choose."""
    assert spoken_configurations(NEW_DIMENSION).count("3 B H K") == 1


def test_bhk_is_spaced_for_the_voice_engine():
    """Written solid, Sarvam tries to pronounce it as a word."""
    for unit in spoken_configurations(NEW_DIMENSION):
        assert "BHK" not in unit
        assert unit.endswith("B H K")


def test_a_villa_or_plot_is_named_whole():
    """No BHK count to collapse on, so the project's own words survive intact."""
    assert spoken_configurations(
        [{"type": "4 BHK Villa"}, {"type": "Villament"}, {"type": "Plot"}]
    ) == ["4 B H K", "Villament", "Plot"]


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        "2 BHK",
        [{}],
        [{"type": "  "}],
        ["2 BHK"],
        # Not iterable at all. Only the isinstance check stops these reaching the for loop,
        # and a TypeError here fires inside the websocket handler — the call dies before the
        # agent says a word, which is far worse than a missing line of context.
        123,
        {"2 BHK": "1.17 Cr"},
    ],
)
def test_a_malformed_config_column_never_kills_the_call(config):
    """This column is hand-filled per project. A shape nobody anticipated must produce no
    line rather than an exception inside the websocket handler."""
    assert spoken_configurations(config) == []


# ─── how it reaches the model ─────────────────────────────────────────────────────────


def _context() -> str:
    return build_campaign_context(
        {
            "name": "Abhee Codename New Dimension",
            "locality": "Varthur",
            "config_json": NEW_DIMENSION,
        }
    )


def test_the_context_hands_over_a_finished_sentence():
    """Given only the six-row table, the model summarised it and got it wrong. The point of
    this line is that there is nothing left to summarise."""
    line = next(
        l for l in _context().splitlines() if l.startswith("Configurations to name")
    )
    assert "2 B H K, 3 B H K, 3.5 B H K, 4.5 B H K" in line
    assert "never round" in line


def test_the_priced_table_is_still_there():
    """The spoken list is for the intro. Prices, areas and trim levels are what answer the
    questions that follow, and losing them would be a worse trade than the bug."""
    context = _context()
    assert "3.5 BHK Presidential" in context
    assert "2.06 Cr" in context


def test_the_prompt_points_at_the_list_and_forbids_rounding():
    prompt = get_system_prompt(_context())
    assert "Configurations to name" in prompt
    assert "NEVER round a configuration" in prompt
