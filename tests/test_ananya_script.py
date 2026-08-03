"""The Ananya call script: greeting, the intent gate, and site-visit availability.

The previous agent opened with "This is Priya calling from X. How are you doing today?",
asked for the name only midway through, pitched before establishing intent, and offered
visits on weekends alone. Each of those cost calls.
"""

import inspect

import pytest

from app.prompts.agent_prompts import get_system_prompt
from app.services.agent import build_opening_line
from app.utils.context_builder import build_campaign_context

CONTEXT = "Project Name: Test\nLocation: Test"
PROMPT = get_system_prompt(CONTEXT)
NAMED = get_system_prompt(CONTEXT, "Rahul Sharma")


# --- greeting ----------------------------------------------------------------------


def test_the_agent_introduces_herself_as_ananya():
    line = build_opening_line("Abhee New Dimension", "Rahul")
    assert line.startswith("Hi, I am Ananya calling you on behalf of Abhee New Dimension.")
    assert "Priya" not in line


def test_the_greeting_confirms_the_name_from_the_dial_list():
    assert build_opening_line("X", "Rahul").endswith("Am I speaking with Rahul?")


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_without_a_name_it_asks_instead_of_inventing_one(missing):
    line = build_opening_line("X", missing)
    assert "Am I speaking with" not in line, "a guessed name is worse than no name"
    assert "good name" in line


def test_the_prompt_carries_the_name_so_a_mid_call_reintroduction_matches():
    """If the prospect speaks first the canned line is cancelled, and the model greets."""
    assert "Rahul Sharma" in NAMED
    assert "Am I speaking with Rahul Sharma?" in NAMED
    assert "do NOT have this prospect's name" in PROMPT


# --- the intent gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Are you looking for any property purchase?",
        "We are launching a new project in",
        "We have launched a project in",
        "Do NOT pitch the project before you ask this",
    ],
)
def test_the_opening_gate_precedes_the_pitch(phrase):
    assert phrase in PROMPT


@pytest.mark.parametrize(
    "status,stage",
    [
        ("Pre Launch", "PRE_LAUNCH"),
        ("pre-launch", "PRE_LAUNCH"),
        ("Upcoming", "PRE_LAUNCH"),
        ("EOI stage", "PRE_LAUNCH"),
        ("Ready to Move", "LAUNCHED"),
        ("Under Construction", "LAUNCHED"),
        (None, "LAUNCHED"),
    ],
)
def test_launch_stage_is_resolved_for_the_agent(status, stage):
    """Left as prose the model had to guess which of the two openings applied."""
    ctx = build_campaign_context({"name": "X", "locality": "Y", "possession_status": status})
    assert f"Launch Stage: {stage}" in ctx


# --- requalification, site visits, cab ---------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "an apartment, a villa, or a plot",
        "for your own stay, or for investment",
        "Which area are you looking in",
        "What budget are you thinking of",
        "When are you planning to buy",
    ],
)
def test_an_uninterested_prospect_is_still_qualified(phrase):
    """A no to this project is not a no to every project — capture the requirement."""
    assert phrase in PROMPT


def test_site_visits_run_on_weekdays_too():
    assert "weekdays AND weekends" in PROMPT
    assert "Never say visits happen only on weekends" in PROMPT


def test_the_visit_booking_captures_a_day_and_a_time():
    assert "specific DAY or date" in PROMPT
    assert "specific TIME between 10 AM and 8 PM" in PROMPT


def test_a_cab_is_offered_only_when_the_project_actually_has_one():
    """Promising a pickup the project does not run is a complaint on the day of the visit."""
    assert "only if the campaign context mentions a cab" in PROMPT
    assert "If the campaign context does not mention it, NEVER offer a cab" in PROMPT
    assert "ask for the pickup location" in PROMPT


# --- simple English ----------------------------------------------------------------


def test_simple_english_is_stated_as_the_top_style_rule():
    """Replies were being written in brochure English that second-language callers on a
    phone line could not follow in one listen."""
    assert "SIMPLE ENGLISH — THE MOST IMPORTANT RULE" in PROMPT
    assert "beats every other style rule" in PROMPT


def test_the_prompt_forbids_speaking_tool_syntax():
    assert "NEVER SPEAK TOOL SYNTAX" in PROMPT
    assert "<function=end_call" in PROMPT


# --- wiring ------------------------------------------------------------------------


def test_the_dial_payload_accepts_a_name_per_number():
    from app.api.routes.campaign import DialRequest

    req = DialRequest(phone_numbers=[{"name": "Rahul", "number": "9876543210"}])
    assert req.phone_numbers[0].name == "Rahul"
    assert req.phone_numbers[0].number == "+919876543210", "E.164 normalisation still applies"


def test_a_bare_number_list_still_dials():
    """Existing integrations post flat strings; they must not start failing."""
    from app.api.routes.campaign import DialRequest

    req = DialRequest(phone_numbers=["9876543210"])
    assert req.phone_numbers[0].number == "+919876543210"
    assert req.phone_numbers[0].name is None


def test_the_name_is_recorded_before_the_dial():
    """Compares the calls themselves, not their positions in the text — a comment naming
    trigger_vobiz_call would otherwise be enough to pass or fail this."""
    import ast

    from app.api.routes import campaign

    tree = ast.parse(inspect.getsource(campaign.dial_campaign_vobiz).lstrip())

    def line_of(target: str) -> int:
        lines = [
            n.lineno
            for n in ast.walk(tree)
            if (isinstance(n, ast.Name) and n.id == target)
            or (isinstance(n, ast.Attribute) and n.attr == target)
        ]
        assert lines, f"{target} is not referenced at all"
        return min(lines)

    assert line_of("remember_customer_name") < line_of("trigger_vobiz_call"), (
        "the name must be stored before the dial, or the greeting has nothing to use"
    )


@pytest.mark.parametrize("route", ["dial_campaign_vobiz", "dial_campaign"])
def test_both_dial_routes_send_the_number_not_the_whole_lead(route):
    """The dashboard route shares DialRequest but had its own loop. Passing the model
    through gave Redis 'Invalid input of type: DialTarget' and Vobiz a JSON error."""
    import ast

    from app.api.routes import campaign as campaign_route
    from app.api.routes import dashboard as dashboard_route

    fn = getattr(campaign_route, route, None) or getattr(dashboard_route, route)
    tree = ast.parse(inspect.getsource(fn).lstrip())
    passed = {
        ast.unparse(arg)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            getattr(n.func, "id", None) in {"remember_dialed_number", "trigger_vobiz_call"}
            or "trigger_vobiz_call" in ast.unparse(n)
        )
        for arg in n.args
    }
    assert "target.number" in passed, f"{route} must dial target.number, not the model"
    assert "target" not in passed, "a DialTarget is not serialisable by redis or httpx"


def test_the_dashboard_dial_records_the_name_too():
    from app.api.routes import dashboard

    assert "remember_customer_name" in inspect.getsource(dashboard.dial_campaign), (
        "dialing from the dashboard would greet every lead as a stranger"
    )


def test_the_call_reads_the_name_back_out_of_redis():
    from app.api.routes import webhook

    assert "customer_name=await recall_customer_name(call_sid)" in inspect.getsource(
        webhook._handle_call
    )


def test_the_agent_greets_with_the_lead_name():
    src = inspect.getsource(__import__("app.services.agent", fromlist=["x"]).run_voice_agent)
    assert "build_opening_line(project_name, customer_name)" in src
    assert "get_system_prompt(campaign_context, customer_name)" in src
