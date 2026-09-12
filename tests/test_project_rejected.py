"""Selling a project to somebody who has already said no to it.

Call f1d9804b, then f556caf9 the next morning after the prompt had been told not to:

    USER  → "Yeah but not in this project."
    ... four qualifying questions, all answered ...
    AGENT → "Okay, a budget of about two Crores works well. In this project we have a
             3.5 BHK Presidential at 2.06 Crores and a 4.5 BHK Presidential at 2.64 Crores."
    USER  → "I said I'm not interested in this project."

"I said" is a prospect repeating themselves, and the call ended on it. The static rule in
step 5 had been written the day before.
"""

import ast
from pathlib import Path

import pytest

from app.utils.project_rejected import BRIEF, rejects_the_project

AGENT_SRC = Path("app/services/agent.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "said",
    [
        "Yeah but not in this project.",
        "Actually I don't want in that area and that project.",
        "I said I'm not interested in this project. I want something in monthly.",
        "not in this project",
        "I am not interested in your project",
        "this project is not for me",
        "is project me nahi",
        "ye project nahi chahiye",
        "इस प्रोजेक्ट में नहीं",
    ],
)
def test_ruling_out_this_project(said):
    assert rejects_the_project(said) is True


@pytest.mark.parametrize(
    "said",
    [
        # Leaving the call. end_call and repeat_request own these; holding somebody who is
        # going is a different and worse bug.
        "not interested",
        "no thank you",
        "I am not interested in buying property",
        # The GUESS this must not touch. Step 5's re-check on "not this area" from somebody
        # who has not been told where it is saved a 1.5 Crore lead, and it stays.
        "not in this area",
        "I do not want this area",
        "not this location",
        # Ordinary conversation.
        "3 BHK",
        "two CR",
        "maybe later",
    ],
)
def test_what_it_must_not_read_as_ruling_out_the_project(said):
    assert rejects_the_project(said) is False


@pytest.mark.parametrize("said", [None, "", "   "])
def test_nothing_said_rules_nothing_out(said):
    assert rejects_the_project(said) is False


# --- what the model is then shown -------------------------------------------------------


def test_the_block_stops_the_pitch_rather_than_the_project():
    """The first version of this said "never mention this project again", and that was wrong
    in the other direction: the campaign IS for this project, and a salesperson who hears a
    requirement it fits and says nothing is no use to anyone.

    What the live turn got wrong was not mentioning it. The prospect had named an area and a
    budget, only the budget fitted, and the price list went back to them as though the area
    had never been said."""
    assert "Stop pitching it" in BRIEF
    for part in ("prices", "configurations", "amenities"):
        assert part in BRIEF, part
    assert "never mention this project again" not in BRIEF.lower()


def test_one_more_mention_is_allowed_and_only_one():
    """Once, in one sentence, after the requirement is known — and then their answer is
    final. Without the ceiling this is just the re-pitch with extra steps."""
    assert "THEN ONCE, AND ONLY IF IT GENUINELY MATCHES" in BRIEF
    assert "ONE sentence" in BRIEF
    assert "never raise it a third time" in BRIEF


def test_a_match_means_their_words_and_not_just_the_convenient_one():
    """The whole failure in one rule: matching on the dimension that suits us and ignoring
    the one they actually named. If their own words cannot go in the sentence, it is not a
    match."""
    assert "the things THEY named line up" in BRIEF
    assert "cannot put their own words in the sentence" in BRIEF


def test_the_block_says_what_to_do_first_and_how_to_finish():
    """A prohibition on its own leaves the model with nothing to say next, and a turn with
    nothing to say is where it reaches for the pitch again."""
    assert "FIND OUT WHAT THEY DO WANT" in BRIEF
    assert "One question at a time" in BRIEF
    assert "read back what you noted" in BRIEF


# --- wired the way ALREADY ASKED is ------------------------------------------------------


def _handler_src(name: str) -> str:
    tree = ast.parse(AGENT_SRC)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == name
    )
    return ast.unparse(node)


def test_the_block_reaches_the_model_through_the_same_path_as_already_asked():
    """Appended to the system message rather than added as its own turn. A system turn
    between the conversation's own would read better and is not something this repository
    controls — the chat template is the provider's, and a context they reject is a dead
    call."""
    src = _handler_src("refresh_asked_brief")
    assert "PROJECT_RULED_OUT" in src
    assert "_project_ruled_out" in src


def test_it_is_shown_the_moment_they_say_it():
    """Not on the next turn. The re-pitch happened one turn after the rejection on one call
    and four turns after it on another, so there is no safe number of turns to wait."""
    src = _handler_src("on_user_turn_stopped")
    assert "rejects_the_project(transcript)" in src
    assert "refresh_asked_brief()" in src


def test_it_is_never_taken_away_again():
    """Nobody un-rules-out a project. A block that came and went would let the pitch back in
    on any turn where the words happened not to match."""
    tree = ast.parse(AGENT_SRC)
    to_false = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_project_ruled_out" for t in n.targets)
        and ast.unparse(n.value) == "False"
    ]
    assert not to_false, "something clears it back to False"
    assert "_project_ruled_out: bool = False" in AGENT_SRC, "it has to start off"
    assert "_project_ruled_out = True" in AGENT_SRC


def test_an_ordinary_call_carries_none_of_it():
    """The prompt is resent in full on every turn. Spending tokens describing a rejection
    that has not happened is the cost this mechanism exists to avoid."""
    src = _handler_src("refresh_asked_brief")
    # ast.unparse normalises quotes, so the empty string reads as '' here
    assert "PROJECT_RULED_OUT if _project_ruled_out else ''" in src
