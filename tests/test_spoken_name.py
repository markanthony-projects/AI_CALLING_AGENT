"""How the agent addresses the person who answered.

The greeting spoke the lead-list field verbatim. Live calls produced:

    "Hi, Good morning RAHUL."                       <- the CRM row was capitalised
    "Hi, Good morning mahantesha."                  <- and this one was not
    "Hi, Good afternoon Abhijit Kumar Singh."

Nobody greets a stranger with their full legal name, and a voice engine reads case as
emphasis. All three are one defect: a database field going straight to a speaker, on the
one line the prospect uses to decide whether this is a person or a machine.
"""

import inspect

import pytest

from app.services.agent import build_opening_line
from app.utils.person_name import spoken_name

PROJECT = "Abhee Codename New Dimension"
DEVELOPER = "Abhee Ventures"


# --- the name itself --------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,said",
    [
        ("Abhijit Kumar Singh", "Abhijit"),
        ("RAHUL", "Rahul"),
        ("mahantesha", "Mahantesha"),
        ("  Priya  ", "Priya"),
        ("Chandan", "Chandan"),
    ],
)
def test_a_person_is_greeted_by_their_first_name(raw, said):
    assert spoken_name(raw) == said


@pytest.mark.parametrize(
    "raw,said",
    [
        ("Dr. Rahul Sharma", "Dr. Rahul"),
        ("dr rahul sharma", "Dr. Rahul"),
        ("Mr ABHIJIT kumar", "Mr. Abhijit"),
        ("Smt. Lakshmi Devi", "Smt. Lakshmi"),
        ("Prof Anand", "Prof. Anand"),
    ],
)
def test_a_salutation_stays_with_the_name_it_belongs_to(raw, said):
    """"Dr. Rahul" is how that person is addressed. "Rahul" is a demotion, and on a sales
    call to someone who put the title in the form it is the wrong foot to start on."""
    assert spoken_name(raw) == said


def test_a_deliberate_capital_inside_a_name_is_left_alone():
    """The case is corrected only when the whole word is one case, because that is when it
    carries no information. "DeSouza" and "McKenna" spell themselves, and title-casing them
    would introduce a mistake while fixing one."""
    assert spoken_name("DeSouza Fernandes") == "DeSouza"
    assert spoken_name("McKenna") == "McKenna"


@pytest.mark.parametrize("raw,said", [("R Kumar", "Kumar"), ("K. S. Sharma", "Sharma")])
def test_an_initial_is_skipped_for_the_word_somebody_is_called_by(raw, said):
    """Indian lead lists carry these as often as they carry a first name first, and
    "Good morning R." is worse than using the next word along."""
    assert spoken_name(raw) == said


@pytest.mark.parametrize("nothing", [None, "", "   ", "A. B.", "R."])
def test_nothing_usable_is_no_name_at_all(nothing):
    """Empty is a real answer, and the greeting already handles it — it leaves the name out
    and the agent asks in its first reply."""
    assert spoken_name(nothing) == ""


@pytest.mark.parametrize("unsayable", ["राहुल", "Rahul123", "+919844014300", "???"])
def test_a_row_the_voice_engine_cannot_read_is_dropped_rather_than_spoken(unsayable):
    """The greeting is queued as a TTSSpeakFrame and never passes the tool-syntax filter, so
    nothing else is checking it. Sarvam breaks up mid-word on mixed script, and a garbled
    name on the opening line is worse than no name at all."""
    assert spoken_name(unsayable) == ""


def test_it_can_be_applied_twice_without_changing_the_answer():
    """It runs once before the prompt and again inside the greeting. Idempotence is what
    makes that safe rather than a bug waiting for a second caller."""
    for raw in ["Abhijit Kumar Singh", "RAHUL", "Dr. Rahul Sharma", "R Kumar", ""]:
        once = spoken_name(raw)
        assert spoken_name(once) == once, raw


# --- and how the call uses it -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,said", [("Abhijit Kumar Singh", "Abhijit"), ("RAHUL", "Rahul"), ("mahantesha", "Mahantesha")]
)
def test_the_greeting_says_it_the_way_a_person_would(raw, said):
    line = build_opening_line(PROJECT, raw, developer_name=DEVELOPER)
    assert f"Good morning {said}." in line or f"Good afternoon {said}." in line or f"Good evening {said}." in line
    assert raw not in line or raw == said


def test_a_greeting_with_no_usable_name_reads_as_a_sentence():
    """Not "Good morning ." — the space and the name go together or neither does."""
    line = build_opening_line(PROJECT, "राहुल", developer_name=DEVELOPER)
    assert " ." not in line
    assert "Hi, Good" in line


def test_the_prompt_is_told_the_same_name_the_greeting_said():
    """Handed the full name, the model uses the full name for the rest of the call — and
    then the opening line and every turn after it disagree about who it is talking to."""
    src = inspect.getsource(
        __import__("app.services.agent", fromlist=["x"]).run_voice_agent
    )
    converted = src.index("customer_name = spoken_name(customer_name)")
    assert converted < src.index("get_system_prompt(campaign_context, customer_name)")
    assert converted < src.index("build_opening_line(")


def test_an_unusable_name_reaches_the_prompt_as_absent_and_not_as_empty():
    """The prompt branches on whether there is a name: without one it tells the model to ask
    for it, and with one it says to greet by it. An empty string is neither — it would take
    the "greet them by it" branch with nothing to greet them by."""
    src = inspect.getsource(
        __import__("app.services.agent", fromlist=["x"]).run_voice_agent
    )
    assert "customer_name = spoken_name(customer_name) or None" in src


# --- the opening the prospect actually hears -----------------------------------------------


def test_the_project_is_not_the_first_thing_said_about_it():
    """From a live call: "We are launching Abhee Codename New Dimension in Varthur -
    Sarjapur Road." A name they have never heard means nothing until they know what it is,
    and hearing it first makes them work out what is being said instead of listening."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "We are launching a new project in [location]." in prompt
    assert "We are launching [project name]" not in prompt
    assert "NEVER open with the project name" in prompt


def test_the_name_still_gets_said_once_they_know_what_it_is():
    """Removing it entirely would be the older bug back: the prospect never learns what the
    project is called."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "It is called [project name] —" in prompt


def test_the_headline_still_rides_with_the_name():
    """The headline is the only reason they have to keep listening. Naming the project
    without it is a label and no argument."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    named = prompt.index("It is called [project name] —")
    assert 'Headline' in prompt[named:named + 200]


def test_the_intent_gate_is_still_the_question_that_ends_the_opening():
    """It decides whether the call goes to the pitch or to step 5, so it has to be a
    question about buying — not "shall I tell you more", which sorts nobody."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    gate = prompt.index("Are you looking for any property purchase?")
    assert prompt.index("It is called [project name]") < gate
