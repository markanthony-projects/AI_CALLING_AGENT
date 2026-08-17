"""The agent must name the configurations the project actually sells.

From call 5664ace6, on Abhee Codename New Dimension:

    AGENT → "We have 2, 3, and 4 BHK homes starting from 1.17 Crores."

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
    assert spoken_configurations(NEW_DIMENSION) == "2, 3, 3.5 and 4.5 BHK"


def test_the_letters_are_said_once_and_not_after_every_number():
    """The first fix produced the second bug. Handed four finished labels, the agent said:

        "We have 2 BHK, 3 BHK, 3.5 BHK, and 4.5 BHK units starting at 1.17 Crores."

    Factually right, and it sounded like a machine — which on a sales call is its own kind
    of wrong. A person says the counts once and the acronym once.
    """
    phrase = spoken_configurations(NEW_DIMENSION)
    assert phrase.count("BHK") == 1
    assert phrase.count(",") == 2  # "2, 3, 3.5 and 4.5" — three commas would be four counts
    assert phrase == "2, 3, 3.5 and 4.5 BHK"


def test_a_half_configuration_is_never_rounded():
    """The original failure. 4.5 became 4, which is a flat the project does not sell."""
    assert spoken_configurations([{"type": "4.5 BHK Presidential"}]) == "4.5 BHK"


def test_nothing_the_project_sells_is_dropped():
    """3.5 disappeared entirely from the spoken list."""
    phrase = spoken_configurations(NEW_DIMENSION)
    for count in ("2", "3", "3.5", "4.5"):
        assert count in phrase


def test_trim_levels_collapse_to_one_configuration():
    """Regular, Comfort and Luxury are one thing to name on an opening call. Reading all
    three back is a brochure, and the full table is still in the context when they choose."""
    assert spoken_configurations(NEW_DIMENSION).count("3,") == 1


def test_the_acronym_is_written_solid():
    """It was spaced to "B H K" on the belief that Sarvam would attack it as a word. Measured
    against bulbul:v3 that is no longer true — the same sentence runs 5.55s solid against
    6.31s spaced — and spelling it out is what made the list sound mechanical."""
    phrase = spoken_configurations(NEW_DIMENSION)
    assert "BHK" in phrase
    assert "B H K" not in phrase


def test_there_is_no_oxford_comma():
    """Plain English. Sarvam does not pause at a comma at all, so it buys nothing on the
    line either."""
    assert ", and" not in spoken_configurations(NEW_DIMENSION)


def test_a_villa_or_plot_is_named_whole():
    """No BHK count to group on, so the project's own words survive — and they come after
    the grouped counts, because the acronym belongs to the numbers in front of it."""
    assert (
        spoken_configurations([{"type": "4 BHK Villa"}, {"type": "Villament"}, {"type": "Plot"}])
        == "4 BHK, Villament and Plot"
    )


def test_a_single_configuration_needs_no_joining():
    assert spoken_configurations([{"type": "Plot"}]) == "Plot"


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
    assert spoken_configurations(config) == ""


# ─── how it reaches the model ─────────────────────────────────────────────────────────


def _context() -> str:
    return build_campaign_context(
        {
            "name": "Abhee Codename New Dimension",
            "locality": "Varthur",
            "config_json": NEW_DIMENSION,
        }
    )


def test_the_context_hands_over_a_finished_phrase():
    """Given only the six-row table the model summarised it and got the facts wrong; given a
    list of labels it got the facts right and read them out like a machine. The point of this
    line is that there is nothing left for it to either summarise or re-join."""
    line = next(l for l in _context().splitlines() if l.startswith("Configurations"))
    assert "2, 3, 3.5 and 4.5 BHK" in line
    assert "never round" in line
    assert "never repeat BHK" in line


def test_the_priced_table_is_still_there():
    """The spoken list is for the intro. Prices, areas and trim levels are what answer the
    questions that follow, and losing them would be a worse trade than the bug."""
    context = _context()
    assert "3.5 BHK Presidential" in context
    assert "2.06 Cr" in context


def test_the_prompt_points_at_the_phrase_and_forbids_rounding():
    prompt = get_system_prompt(_context())
    assert "Configurations" in prompt
    assert "NEVER round a configuration" in prompt


def test_the_prompt_forbids_re_expanding_the_phrase():
    """Naming the phrase is not enough on its own — expanding a list feels to a model like
    being helpfully explicit, not like undoing a fix.

    The worked example lives in exactly one place. It used to appear in both the SPEAKING
    STYLE rule and the UNIT TYPES step, and every duplicated word here is resent on every
    turn of every call.
    """
    prompt = get_system_prompt(_context())
    assert prompt.count('never "2 BHK, 3 BHK, 3.5 BHK and 4.5 BHK"') == 1
    style = next(l for l in prompt.splitlines() if l.startswith("- Do NOT stack the same word"))
    assert 'never "2 BHK, 3 BHK, 3.5 BHK and 4.5 BHK"' in style

    unit_types = next(l for l in prompt.splitlines() if "UNIT TYPES:" in l)
    assert "word for word" in unit_types
    assert "NEVER round a configuration" in unit_types
