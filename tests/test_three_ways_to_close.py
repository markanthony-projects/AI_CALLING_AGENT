"""The call had one close, and it chased it.

On 3 Sep the agent asked "Would you like to visit the site?" nineteen times in one call.
The prospect said "Apart from visit, you are not telling any details", then "No, I'm not
interested", and a three-crore lead was gone.

That was not stubbornness. A site visit was the only close the script knew, so once
qualifying was done there was nowhere else to go — and a rule saying "stop asking" would
have left it with nothing to say instead. The fix is to give it somewhere else to land.

The other half is what comes before the close. The project used to be described in one
reply capped at 35 words, and the model filled 32 of them — eleven seconds of speaking with
the prospect silent throughout. Now it is four short turns with a question on the end of
each, so they have spoken four times before anyone mentions a visit.
"""

import pytest

from app.prompts.agent_prompts import get_system_prompt

PROMPT = get_system_prompt("Project Name: Test\nLocation: Test", "Rahul")
CLOSE = PROMPT[PROMPT.index("THE CLOSE:") : PROMPT.index("NEVER ASK THE SAME THING TWICE")]


# --- somewhere else to land -----------------------------------------------------------------


def test_the_close_no_longer_calls_itself_the_site_visit():
    """The heading was "SITE VISIT AND THE CLOSE", which told the model those were the same
    thing. They are not, and on one call that cost a lead."""
    assert "THE CLOSE:" in PROMPT
    assert "SITE VISIT AND THE CLOSE" not in PROMPT


@pytest.mark.parametrize("exit_", ["WHATSAPP:", "CALLBACK:"])
def test_there_is_a_way_out_that_is_not_a_visit(exit_):
    assert exit_ in CLOSE


def test_the_visit_is_offered_once_and_never_again():
    assert "Offer the visit ONCE" in CLOSE
    assert "do NOT ask again" in CLOSE


def test_a_hesitation_counts_as_a_no():
    """"Apart from visit, you are not telling any details" is not a yes. Waiting for a flat
    refusal before trying anything else is how the nineteen happened."""
    assert "say no, hesitate, or change the subject" in CLOSE


def test_the_model_is_told_which_ending_is_a_good_one():
    """Otherwise a call that ends in a WhatsApp send reads as a failed visit."""
    assert "details on their phone is a good call" in CLOSE
    assert "same question asked four times is a lost one" in CLOSE


def test_the_visit_is_only_offered_after_they_have_shown_interest():
    """Offering it in the opening is the same chase, earlier."""
    assert "only after they have shown interest in something specific" in CLOSE


# --- and the rules that were already earned stay earned ---------------------------------------


@pytest.mark.parametrize(
    "phrase,failure",
    [
        ("NOT THE END OF THE CALL", "hung up 457ms after the prospect agreed to a visit"),
        ("read it back", "a booking the prospect was never told about"),
        ("does not turn up", "why the read-back is not optional"),
        ("weekdays AND weekends", "told a prospect visits happen only at weekends"),
        ("specific DAY or date", "\"this weekend\" stored as a booking"),
        ("If the campaign context does not mention it, NEVER offer a cab", "a pickup we do not run"),
        ("there are no visiting hours", "an hour refused that the team would have taken"),
    ],
)
def test_a_rewrite_of_the_close_did_not_drop_a_rule(phrase, failure):
    """Every one of these was added after a live call went wrong. Rewriting the section
    around them is fine; losing one of them is how the same call happens twice."""
    assert phrase in PROMPT, f"lost the rule guarding: {failure}"


def test_the_repeat_rule_still_points_at_the_block_that_carries_the_detail():
    """The prose was cut down because ALREADY ASKED supplies the specifics at runtime. The
    pointer to it has to survive the cut, or the block arrives unannounced."""
    assert "ALREADY ASKED" in PROMPT
    assert "follow it exactly" in PROMPT
    assert "send the details on WhatsApp and call end_call" in PROMPT


# --- the script it replaced -------------------------------------------------------------------


def test_the_old_script_is_kept_where_it_can_be_found():
    """Asked for it back, "look through git history" means knowing which commit. Most of that
    script is unchanged in this one — only the flow and the close were rewritten — so the
    copy is there to compare against, not only to restore."""
    import pathlib

    archived = pathlib.Path("app/prompts/archive/2026-09-07-site-visit-led.txt")
    assert archived.exists()
    text = archived.read_text(encoding="utf-8")
    assert "SHORT INTRO" in text, "the archive does not contain the flow it was kept for"
    assert "SITE VISIT AND THE CLOSE" in text
    assert "WHY IT WAS REPLACED" in text, "a copy with no reason attached is a mystery later"


def test_the_archive_cannot_be_imported_by_accident():
    """A .py beside the live prompt is one careless import away from being the live prompt."""
    import pathlib

    stray = list(pathlib.Path("app/prompts/archive").glob("*.py"))
    assert not stray, stray
