"""The call script: greeting, the intent gate, and site-visit availability.

An earlier agent asked for the name only midway through, pitched before establishing
intent, and offered visits on weekends alone. Each of those cost calls.
"""

import inspect
from datetime import datetime

import pytest

from app.prompts.agent_prompts import AGENT_NAME, get_system_prompt
from app.services.agent import build_opening_line
from app.utils.context_builder import build_campaign_context, pitch_points

# Naive UTC, as utc_now() produces. 04:00 UTC is 09:30 IST — the start of a dialing shift.
MORNING = datetime(2026, 8, 13, 4, 0)
AFTERNOON = datetime(2026, 8, 13, 8, 0)  # 13:30 IST
EVENING = datetime(2026, 8, 13, 13, 0)  # 18:30 IST

CONTEXT = "Project Name: Test\nLocation: Test"
PROMPT = get_system_prompt(CONTEXT)
NAMED = get_system_prompt(CONTEXT, "Rahul Sharma")


# --- greeting ----------------------------------------------------------------------


def test_the_whole_greeting():
    assert build_opening_line("Abhee New Dimension", "Rahul", MORNING) == (
        "Hi, Good morning Rahul. I am Priya calling you from Abhee New Dimension. "
        "Can I speak to you for a minute?"
    )


@pytest.mark.parametrize(
    "when, part", [(MORNING, "morning"), (AFTERNOON, "afternoon"), (EVENING, "evening")]
)
def test_the_greeting_follows_the_prospects_clock(when, part):
    """The droplet runs on UTC. Read there, 09:30 IST — the middle of a dialing shift —
    looks like 04:00, and the prospect is wished good morning at what the machine thinks is
    the dead of night. The 5h30m offset moves the afternoon boundary by half a day too."""
    assert f"Good {part} " in build_opening_line("X", "Rahul", when)


def test_it_asks_for_a_minute_rather_than_interrogating_them():
    """"Am I speaking with Rahul?" opens by making them account for themselves. Using the
    name to address them and asking for a minute is how a person opens a call."""
    line = build_opening_line("X", "Rahul", MORNING)
    assert line.endswith("Can I speak to you for a minute?")
    assert "Am I speaking with" not in line


def test_the_agent_and_the_prompt_agree_on_who_is_calling():
    """The greeting is played by the system, but it is cancelled if the prospect speaks
    first and then the MODEL introduces itself. Two copies of the name would drift, and the
    caller would be handed to a different person mid-call."""
    assert AGENT_NAME in build_opening_line("X", "Rahul", MORNING)
    assert AGENT_NAME in PROMPT


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_without_a_name_the_greeting_omits_it_rather_than_guessing(missing):
    line = build_opening_line("X", missing, MORNING)
    assert line.startswith("Hi, Good morning. I am")
    assert "None" not in line and "  " not in line


def test_without_a_name_the_model_is_told_to_ask_in_its_first_reply():
    """The greeting no longer asks, so nothing else will. A call that runs to the end with
    no name produces a lead sales cannot follow up properly.

    Scoped to the NAME paragraph on purpose: "VERY FIRST reply" also appears in the GREETING
    step, so searching the whole prompt passes even when this instruction is gone.
    """
    name_rule = next(l for l in PROMPT.splitlines() if l.startswith("NAME:"))
    assert "do NOT have this prospect's name" in name_rule
    assert "VERY FIRST reply" in name_rule
    assert "good name" in name_rule


def test_the_name_question_is_the_whole_of_that_reply():
    """On call 2afae4e1 the agent said, in one breath:

        "May I know your good name? Also, we are launching a new project in Varthur. It is
        Bengaluru's first Scotland-themed residential township. Are you looking for any
        property purchase?"

    Two questions, and the prospect hung up. "Ask in your very first reply, before anything
    else" was read as an instruction about word order inside the sentence. It has to say
    that the question IS the reply.
    """
    name_rule = next(l for l in PROMPT.splitlines() if l.startswith("NAME:"))
    assert "ONLY this, and nothing else at all" in name_rule
    assert "Do NOT add the project" in name_rule
    assert "wait for them to answer" in name_rule


def test_one_question_per_reply_is_a_rule_for_the_whole_call():
    """It was only ever stated inside step 4, so the opening and the requalification both
    doubled up. The same call also asked "Which area are you looking in, and when are you
    planning to buy?" — on a phone line the prospect answers one and the other is lost."""
    assert "ONE question per reply, always" in PROMPT
    style = PROMPT[PROMPT.index("SPEAKING STYLE:") :]
    assert "ONE question per reply" in style, "the rule must sit outside the DISCOVERY step"


def test_with_a_name_the_model_addresses_rather_than_verifies():
    assert "Rahul Sharma" in NAMED
    assert "not asking them to prove who they are" in NAMED
    # Reaching the wrong person is still handled, just not as the opening move.
    assert "If they say it is someone else" in NAMED


# --- the opening hook --------------------------------------------------------------


PROJECT = {
    "name": "Abhee Codename New Dimension",
    "locality": "Varthur",
    "usps": [
        "Bengaluru's first Scotland-themed residential township",
        "45-acre golf township with an on-site 3-acre golf course",
        "Pre-launch EOI pricing, around 20 to 30 Lakhs below the expected launch price",
        "RERA registered",
    ],
}


def test_the_headline_is_the_first_curated_selling_point():
    """usps is hand-written per project in selling order, so the first entry is the headline
    by construction. Asking the model to pick "the most attractive" from a dozen bullets
    made it choose fresh on every call, and it chose differently each time."""
    headline, _ = pitch_points(PROJECT)
    assert headline == "Bengaluru's first Scotland-themed residential township"


def test_the_money_argument_is_pulled_out_of_the_hook():
    """"Twenty to thirty Lakhs below the launch price" answers a question the prospect has
    not asked in the first ten seconds. Next to the price it is the reason to keep
    listening; in the opening it is noise."""
    headline, benefit = pitch_points(PROJECT)
    assert "EOI" in benefit
    assert "EOI" not in headline


@pytest.mark.parametrize(
    "usp",
    [
        "Pre-launch EOI pricing, 20 Lakhs below the expected launch price",
        "Introductory pricing for the first 50 buyers",
        "Early-bird offer closing this month",
        "Prices are lower than the launch price",
    ],
)
def test_price_benefits_are_recognised_however_they_are_worded(usp):
    _, benefit = pitch_points({"usps": ["Sea facing towers", usp]})
    assert benefit == usp


def test_the_discount_is_not_also_used_as_the_hook():
    """Builders often list the offer first, because it is what they most want said. It is
    still the wrong opening line — and if it were used as both, the prospect would hear the
    same sentence twice, once in the gate and once with the price."""
    headline, benefit = pitch_points(
        {"usps": ["Pre-launch EOI pricing, 20 Lakhs below launch", "45-acre golf township"]}
    )
    assert benefit == "Pre-launch EOI pricing, 20 Lakhs below launch"
    assert headline == "45-acre golf township"


def test_a_blank_entry_is_never_the_headline():
    """usps is hand-filled, and a stray empty string in the JSON would otherwise become the
    opening hook — the agent would say the launch stage and then nothing at all.

    The price benefit has to be present for this to bite: without one the headline filter is
    comparing against "" and drops blanks by accident. With one it compares against the
    discount instead, and the blank sails through.
    """
    headline, benefit = pitch_points(
        {"usps": ["", "   ", "Pre-launch EOI pricing, 20 Lakhs below launch", "Sea facing towers"]}
    )
    assert benefit == "Pre-launch EOI pricing, 20 Lakhs below launch"
    assert headline == "Sea facing towers"


def test_a_project_with_no_price_benefit_still_gets_a_headline():
    """Most launched projects have no EOI discount. That must not cost them their hook."""
    headline, benefit = pitch_points({"usps": ["Sea facing towers", "RERA registered"]})
    assert headline == "Sea facing towers"
    assert benefit == ""


@pytest.mark.parametrize("usps", [None, [], "not a list", ["", "   "]])
def test_a_project_with_no_usps_never_breaks_the_call(usps):
    """This column is hand-filled. A shape nobody anticipated must cost a line of context,
    not raise inside the websocket handler before the agent has said a word."""
    assert pitch_points({"usps": usps}) == ("", "")


def test_both_reach_the_model_as_named_lines():
    ctx = build_campaign_context(PROJECT)
    headline = next(l for l in ctx.splitlines() if l.startswith("Headline"))
    benefit = next(l for l in ctx.splitlines() if l.startswith("Price benefit"))
    assert "Scotland-themed" in headline
    assert "20 to 30 Lakhs" in benefit
    assert "never before it" in benefit, "the discount must not precede the price"
    # The full list stays: the hook is for the opening, the rest answers what follows.
    assert "RERA registered" in ctx


def test_the_prompt_puts_the_hook_in_the_opening_and_the_money_with_the_price():
    gate = PROMPT[PROMPT.index("2. OPENING GATE") : PROMPT.index("3. SHORT INTRO")]
    assert "Headline" in gate
    assert "Price benefit" not in gate, "the discount belongs next to the price, not before"
    intro = PROMPT[PROMPT.index("3. SHORT INTRO") : PROMPT.index("4. DISCOVERY")]
    assert "Price benefit" in intro


# --- the intent gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Are you looking for any property purchase?",
        "We are launching a new project in",
        "We have launched a project in",
        "Do NOT list amenities, prices or configurations before you ask this",
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


def test_the_api_dial_route_sends_the_number_not_the_whole_lead():
    """Passing the model through gave Redis 'Invalid input of type: DialTarget' and Vobiz a
    JSON error. This route still dials directly; the dashboard one now queues."""
    import ast

    from app.api.routes import campaign as campaign_route

    tree = ast.parse(inspect.getsource(campaign_route.dial_campaign_vobiz).lstrip())
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
    assert "target.number" in passed, "must dial target.number, not the model"
    assert "target" not in passed, "a DialTarget is not serialisable by redis or httpx"


def test_the_dashboard_dial_queues_instead_of_calling():
    """It used to add one background task per number and let the concurrency cap be checked
    later, when each websocket opened — after the carrier had billed us and rung a real person.
    It now writes contacts and lets the pump place the calls a slot at a time."""
    from app.api.routes import dashboard

    src = inspect.getsource(dashboard.dial_campaign)
    assert "trigger_vobiz_call" not in src, "the dashboard still dials without taking a slot"
    assert "Contact" in src and "on_conflict_do_nothing" in src


def test_the_queued_name_survives_to_the_call():
    """The greeting addresses the prospect by name, so the name has to travel with the number.
    On the queued path it is a column rather than a Redis key, and the pump reads it back."""
    from app.api.routes import dashboard
    from app.services import dial_pump

    assert '"name": target.name' in inspect.getsource(dashboard.dial_campaign)
    assert "remember_customer_name(call_sid, contact.name)" in inspect.getsource(dial_pump._place)


def test_queueing_the_same_list_twice_does_not_reset_anybody():
    """An operator unsure whether the first paste worked will paste again. Somebody already
    called must not go back to PENDING and be dialled a second time."""
    from app.api.routes import dashboard

    src = inspect.getsource(dashboard.dial_campaign)
    assert "on_conflict_do_nothing(constraint=\"uq_contacts_campaign_phone\")" in src


def test_the_dashboard_dial_honours_the_do_not_call_list():
    """The lookup has to be against the numbers being queued. A grep for the column name
    passes just as well when the predicate is `where(False)`, which suppresses nobody."""
    import ast

    from app.api.routes import dashboard

    src = inspect.getsource(dashboard.dial_campaign)
    assert "ContactStatus.DND" in src
    tree = ast.parse(src.lstrip())
    lookups = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "where"
        and "Suppression.phone_number" in ast.unparse(n)
    ]
    assert lookups, "the do-not-call list is never queried"
    assert any("in_(numbers)" in q for q in lookups), (
        f"the lookup does not test the numbers being queued: {lookups}"
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
