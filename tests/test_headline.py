"""The headline as a sentence, decided in code because the prompt could not decide it.

    fe1e1c7  "THAT LINE MUST HAVE A VERB IN IT"          -> worked on one call
    44f3951  + "two sentences must not start the same"   -> verb dropped to satisfy it
    781077a  + "vary by rewriting, never by deleting"    -> still a caption

Three attempts, and the third made it worse. By then one paragraph of the prompt carried
seven rules about one sentence. The model was not careless; it was over-constrained.
"""

import pytest

from app.utils.headline import as_a_sentence


@pytest.mark.parametrize(
    "label,spoken",
    [
        # The live one, from call fe961761 onward.
        (
            "Bengaluru's first Scotland-themed residential township",
            "It is Bengaluru's first Scotland-themed residential township.",
        ),
        ("India's first themed township", "It is India's first themed township."),
        ("45 acres of open space", "It is 45 acres of open space."),
    ],
)
def test_a_label_becomes_something_a_person_could_say(label, spoken):
    assert as_a_sentence(label) == spoken


@pytest.mark.parametrize(
    "label,spoken",
    [
        ("A Scotland-themed township", "It is a Scotland-themed township."),
        ("An award-winning township", "It is an award-winning township."),
        ("The largest gated community in East Bangalore",
         "It is the largest gated community in East Bangalore."),
    ],
)
def test_a_leading_article_is_lowered(label, spoken):
    """"It is A Scotland-themed township." reads as a mistake. Only the article moves —
    a name that happens to lead the line keeps its capital, because there is no way to tell
    "Luxury homes" from "Prestige homes" and guessing wrong on a brand is the worse error."""
    assert as_a_sentence(label) == spoken


@pytest.mark.parametrize(
    "already",
    [
        "It is Bengaluru's first golf-course township",
        "This project has a 3-acre golf course",
        "The project offers a private lake",
        "We bring Scotland to Varthur",
        "Our homes are built around a lake",
    ],
)
def test_copy_that_is_already_a_sentence_is_left_alone(already):
    """A second "It is" bolted on the front would be worse than the label ever was, and
    somebody wrote these deliberately."""
    assert as_a_sentence(already) == already + "."


@pytest.mark.parametrize(
    "label",
    ["Luxury homes with a private lake.", "A township", "The lake."],
)
def test_it_never_ends_in_two_full_stops(label):
    assert not as_a_sentence(label).endswith("..")
    assert as_a_sentence(label).endswith(".")


@pytest.mark.parametrize("nothing", [None, "", "   ", ".", " . ", ",,"])
def test_nothing_usable_produces_nothing(nothing):
    """So the caller leaves the line out altogether rather than handing the model "It is."."""
    assert as_a_sentence(nothing) == ""


def test_the_same_input_always_produces_the_same_line():
    """The part no prompt can promise, and the reason this moved into code at all."""
    label = "Bengaluru's first Scotland-themed residential township"
    assert len({as_a_sentence(label) for _ in range(20)}) == 1
