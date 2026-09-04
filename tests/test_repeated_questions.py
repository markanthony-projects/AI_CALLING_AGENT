"""The agent asking the same thing until the prospect hangs up.

Live call, 3 Sep 2026. "Would you like to visit the site and see it once?" — nineteen times.

    "Apart from visit, you are not telling any details."
    "No, I'm not interested."

Three crores, lost to one question. The model has no memory beyond the transcript it
re-reads every turn, and under 2,686 words of rules it stops telling what it has already
done from what it is about to do.

Only what WE said is tracked. That needs no model and cannot be wrong. What the prospect
ANSWERED needs an extractor, can go stale, and a confidently wrong "you told me six months"
is worse than no memory at all — so it is deliberately not here.

The half that is easy to forget is the way out. Telling a model "stop asking that" and
nothing else leaves it with no move, and on that call the repetition WAS the absence of a
move: the script's only close is a site visit. So every topic carries what to do instead.
"""

import ast
import inspect

import pytest

from app.utils.asked import REPEAT_LIMIT, TOPICS, AskedSoFar, topic_of

VISIT = "That is a good choice. Would you like to visit the site and see it once?"
BUDGET = "No problem at all. What budget are you thinking of?"


# --- reading our own questions ------------------------------------------------------------


@pytest.mark.parametrize(
    "line,key",
    [
        (VISIT, "site_visit"),
        ("Sure. Would you like to come and see it once?", "site_visit"),
        (BUDGET, "budget"),
        ("Got it. When are you planning to buy?", "timeline"),
        ("That is nice. Is this for your own stay, or for investment?", "purpose"),
        ("No problem at all. Which area are you looking in?", "location"),
        ("Sure. Are you looking for an apartment, a villa, or a plot?", "unit_type"),
        ("Of course. Should I call at 6 PM today, or tomorrow at 11 AM?", "callback"),
        ("May I know your good name?", "name"),
    ],
)
def test_it_knows_what_the_agent_just_asked_about(line, key):
    topic = topic_of(line)
    assert topic is not None, line
    assert topic.key == key


@pytest.mark.parametrize(
    "line",
    [
        "That is a great choice for investment.",
        "Prices start at 1.17 Crores.",
        "Thank you for your time. Have a great day!",
        "",
        None,
    ],
)
def test_a_turn_that_asked_nothing_counts_as_nothing(line):
    assert topic_of(line) is None


def test_a_statement_mentioning_a_topic_is_not_a_question_about_it():
    """Only the question at the end of the turn is read. "That is within your budget." is an
    acknowledgement, and counting it would report a repeat that never happened."""
    assert topic_of("That is well within your budget. Which area are you looking in?").key == "location"
    assert topic_of("That is well within your budget.") is None


# --- and what it does with them ------------------------------------------------------------


def test_asking_once_says_nothing_at_all():
    """The conversation working. The prompt is resent in full every turn, and there is no
    room to spend tokens describing a problem that has not happened."""
    asked = AskedSoFar()
    asked.record(VISIT)
    assert asked.brief() == ""


def test_the_second_ask_is_when_it_speaks_up():
    """Once, they answer. Twice, the first did not land — and the third is the pattern that
    lost the call."""
    asked = AskedSoFar()
    asked.record(VISIT)
    asked.record(VISIT)
    assert "a site visit" in asked.brief()


def test_the_block_says_how_many_times():
    asked = AskedSoFar()
    for _ in range(3):
        asked.record(VISIT)
    assert "asked 3 times" in asked.brief()


def test_every_topic_carries_a_way_out():
    """The load-bearing half. "Do not ask that again" with no alternative is a rule that gets
    broken or answered with silence, because the repetition was the missing move."""
    for topic in TOPICS:
        assert topic.instead.strip(), topic.key
        assert len(topic.instead.split()) >= 4, topic.key


def test_the_way_out_is_shown_beside_the_thing_to_stop_asking():
    asked = AskedSoFar()
    for _ in range(2):
        asked.record(VISIT)
    brief = asked.brief()
    assert "WhatsApp" in brief and "end_call" in brief


def test_the_block_tells_the_model_not_to_ask_rather_than_merely_reporting():
    """A list of counts is data. The model needs an instruction."""
    asked = AskedSoFar()
    for _ in range(2):
        asked.record(VISIT)
    assert "Do NOT ask it again" in asked.brief()


def test_topics_asked_once_stay_out_of_the_block():
    """Naming everything asked so far would tell the model to stop doing its job."""
    asked = AskedSoFar()
    asked.record(BUDGET)
    for _ in range(2):
        asked.record(VISIT)
    brief = asked.brief()
    assert "a site visit" in brief
    assert "their budget" not in brief


def test_two_over_asked_topics_both_appear():
    asked = AskedSoFar()
    for line in (VISIT, VISIT, BUDGET, BUDGET):
        asked.record(line)
    brief = asked.brief()
    assert "a site visit" in brief and "their budget" in brief


def test_the_counts_are_readable_for_the_log():
    asked = AskedSoFar()
    asked.record(VISIT)
    asked.record(BUDGET)
    assert asked.counts == {"site_visit": 1, "budget": 1}


def test_the_limit_is_two_and_not_one():
    """Asking a second time after a genuine miss is normal. Reporting at one would fire on
    almost every call and the block would stop meaning anything."""
    assert REPEAT_LIMIT == 2


# --- and how the call wires it up ------------------------------------------------------------


def _agent_source():
    from app.services import agent

    return inspect.getsource(agent.run_voice_agent)


def test_every_agent_turn_is_recorded():
    src = _agent_source()
    assert "asked.record(content)" in src


def test_the_block_reaches_the_model():
    """A tracker nothing shows to the model is a counter, not a fix.

    Asserted as a CALL and not as a substring: the definition line contains the same text,
    so grepping the source passes just as happily with the call site deleted.
    """
    tree = ast.parse(_agent_source().lstrip())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "refresh_asked_brief"
    ]
    assert calls, "the block is built and never shown"


def test_it_is_appended_to_the_system_turn_rather_than_inserted_mid_conversation():
    """A system turn between the conversation's own would have better recency, but Gemma's
    chat template is not something this repository controls, and a context the provider
    rejects is a dead call. Worse recency is a worse call; a rejected context is no call."""
    src = _agent_source()
    tree = ast.parse(src.lstrip())
    targets = [
        ast.unparse(t)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if "messages[" in ast.unparse(t)
    ]
    assert targets == ["messages[0]['content']"], targets


def test_the_block_is_taken_away_again_if_it_stops_applying():
    """It is rebuilt from the counts each time rather than appended to, so the system turn
    cannot accumulate stale copies of itself across a long call."""
    src = _agent_source()
    assert "if brief == _asked_shown:" in src


def test_the_static_prompt_is_never_lost_when_the_block_goes_in():
    """Overwriting the system turn with the block alone would drop every rule the agent has —
    the simple English rule, the tool discipline, the campaign facts, all of it, for the rest
    of the call.

    Asserted on the shape of the expression rather than on its text, because "system_prompt
    appears somewhere in there" is also true of the version that throws it away: it survives
    as the else branch while the block replaces everything on the other side.
    """
    tree = ast.parse(_agent_source().lstrip())
    written = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any("messages[" in ast.unparse(t) for t in n.targets)
    ]
    assert len(written) == 1, [ast.unparse(v) for v in written]
    value = written[0]

    assert isinstance(value, ast.IfExp), ast.unparse(value)
    assert ast.unparse(value.test) == "brief"
    # With no block, the prompt is exactly what it always was.
    assert ast.unparse(value.orelse) == "system_prompt"
    # With one, both are present — not the block on its own.
    joined = ast.unparse(value.body)
    assert "system_prompt" in joined and "brief" in joined, joined
    assert ".join(" in joined, joined


def test_a_model_that_ignores_the_block_is_visible_in_the_log():
    """The interesting failure. Being told and carrying on anyway is what says this approach
    is not working, and it has to be countable across calls rather than replayed by ear."""
    src = _agent_source()
    assert "despite being" in src
    assert "logger.warning" in src[src.index("despite being") - 400 : src.index("despite being")]


def test_the_prompt_tells_the_model_the_block_exists():
    """Injected text the prompt never mentions reads as noise. The rule and the data have to
    refer to each other or the model has no reason to treat it as an instruction."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "ALREADY ASKED" in prompt
    assert "nineteen times" in prompt


def test_the_prompt_has_a_move_for_when_it_runs_out_of_moves():
    """Same reason each topic carries one. Without it the model repeats or goes quiet."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "if you cannot think of one" in prompt
    assert "WhatsApp" in prompt
