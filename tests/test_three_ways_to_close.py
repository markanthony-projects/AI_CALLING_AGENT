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


# --- warm rather than merely short ------------------------------------------------------------


ACKNOWLEDGE = PROMPT[PROMPT.index("ACKNOWLEDGE BEFORE YOU ASK") : PROMPT.index("NEVER JUDGE")]


def test_the_reaction_has_to_be_about_what_they_said():
    """From a live call on 7 Sep, reported as cold: "That works well." opened two different
    replies, and "That is good to know." opened a third. The list of canned reactions was
    being used as a slot to fill rather than as a reaction."""
    assert "THE REACTION MUST BE ABOUT WHAT THEY ACTUALLY SAID" in ACKNOWLEDGE
    assert '"That works well" fits any answer to any question' in ACKNOWLEDGE


def test_the_same_reaction_may_not_be_used_twice():
    """One repetition is what makes a whole conversation sound automated — more than any
    single sentence in it does."""
    assert "NEVER USE THE SAME REACTION TWICE IN ONE CALL" in ACKNOWLEDGE


def test_the_examples_show_the_specific_form_beside_the_generic_one():
    """A rule saying "be specific" with generic examples underneath teaches the examples."""
    assert 'NOT "That works well."' in ACKNOWLEDGE
    assert "Six months is a comfortable time to plan this." in ACKNOWLEDGE


def test_warmth_is_not_loudness():
    """The old rule already banned the showy words. Asking for warmth without repeating that
    is how "excellent!" comes back."""
    assert "Warm is not loud" in ACKNOWLEDGE
    assert "excellent" in ACKNOWLEDGE


def test_the_budget_reaction_stays_neutral_while_the_others_get_specific():
    """The non-judging rule is the one place where generic is correct. Naming the amount at
    all — high or low — is what ends the relationship."""
    assert "Never label the amount, high or low." in ACKNOWLEDGE


def test_the_visit_is_invited_with_a_reason_rather_than_asked_for():
    """"That works well. Would you like to come and see it once?" was reported as
    unprofessional, and it is: the reason is what turns a form question into an invitation."""
    assert "Give a REASON first, tied to something they told you" in CLOSE
    assert "is a question a form asks" in CLOSE


# --- a prospect who talks themselves back in --------------------------------------------------


FLOW = PROMPT[PROMPT.index("5. NOT FOR THEM") : PROMPT.index("OBJECTIONS:")]


def test_a_wrong_area_is_checked_before_it_is_believed():
    """Call 220ce45f, 7 Sep. "Yeah, but not in this area" sent the call to step 5. Four turns
    later the prospect asked for Sarjapur Road — this project's own address — the agent
    replied "Sarjapur Road is a very active area for growth", and then hung up on them.
    Budget 1.5 Crores, buying for investment, inside two months."""
    assert "FIRST, CHECK THEY ARE ACTUALLY NOT FOR US" in FLOW
    assert "they are guessing" in FLOW


def test_there_is_a_way_back_to_the_pitch():
    """Step 5 was one-way. Everything in it led to end_call, so a prospect who ruled
    themselves out by mistake could not be let back in."""
    assert "GO BACK TO STEP 3" in FLOW


@pytest.mark.parametrize(
    "case", ["the area this project is in", "their budget turns out to fit", "unit type is one we sell"]
)
def test_each_way_back_in_is_named(case):
    """"Not this area" is the one that happened, but a budget quoted before they heard the
    price, and a unit type guessed at, rule people out the same way."""
    assert case in FLOW


def test_the_call_that_produced_the_rule_is_written_beside_it():
    """A rule with its own failure attached survives a later rewrite; one without gets
    tidied away by whoever is shortening the prompt next."""
    assert "Sarjapur Road" in FLOW
    assert "hung up on them" in FLOW


def test_giving_up_is_still_possible():
    """The point is to check first, not to never stop. A prospect who really is not for us
    still gets their questions answered and a warm ending."""
    assert "Only when they really are not for us" in FLOW
    assert "never hang up straight away" in FLOW


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
