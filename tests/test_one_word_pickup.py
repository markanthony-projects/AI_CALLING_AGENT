"""The call where the prospect said "Hello" and nothing else.

    Agent:    Hi, Good evening Hari Shanker Choudhary. I am Priya calling you from Abhee
              Codename New Dimension. Can I speak to you for a minute?
    Prospect: Hello.
    Agent:    We are launching a new project in Varthur... Are you looking for any property
              purchase?

It produced a WARM lead carrying the prospect's full name, and marked the contact as having
spoken with the agent so it would never be dialled again. Nothing was learned on that call.
Three separate things had to go wrong at once, and each of them is here.
"""




from app.utils.attribution import name_spoken_by_prospect, prospect_text
from app.worker import _build_system_prompt, resolve_customer_name
from app.utils.timeutils import utc_now

TRANSCRIPT = (
    "Agent: Hi, Good evening Hari Shanker Choudhary. I am Priya calling you from Abhee "
    "Codename New Dimension. Can I speak to you for a minute?\n"
    "Prospect: Hello.\n"
    "Agent: We are launching a new project in Varthur. It is Bengaluru's first Scotland "
    "themed residential township. Are you looking for any property purchase?"
)


# --- what counts as having answered ------------------------------------------------------


def test_only_the_prospect_counts_as_having_spoken():
    """answered_words counted the whole transcript, agent lines included. The agent always
    speaks, so the count was never zero, and record_outcome's "connected but no conversation"
    branch could not run — every call that reached audio was filed as spoke-with-the-agent."""
    assert len(prospect_text(TRANSCRIPT).split()) == 1


def test_a_call_where_the_prospect_never_spoke_counts_as_zero():
    """The case that made the dead branch obvious: nobody said anything and the transcript
    was still full of words."""
    agent_only = "Agent: Hello? Are you there?\nAgent: I will try again later."
    assert len(prospect_text(agent_only).split()) == 0


def test_a_real_conversation_still_counts():
    """The fix must not make every call look unanswered — that would retry people who have
    already spoken to us."""
    talked = TRANSCRIPT + "\nProspect: Yes, I am looking for a 3 BHK around Whitefield."
    assert len(prospect_text(talked).split()) > 5


# --- what counts as a lead ----------------------------------------------------------------


def test_the_prompt_has_a_third_outcome():
    """The rule was two-sided: true for any interest, false only for spam, wrong number,
    telemarketer or explicit refusal. "Hello" is in neither list, and because the false side
    was a closed list, everything in between drifted to true. The model followed the prompt."""
    prompt = _build_system_prompt(utc_now())
    assert "three outcomes" in prompt


def test_the_prompt_says_silence_is_not_interest():
    """The specific reasoning the model has to be given: not refusing is not agreeing."""
    prompt = _build_system_prompt(utc_now())
    assert "Silence is not interest" in prompt
    assert "Not refusing is not agreeing" in prompt


def test_the_prompt_still_admits_a_mismatched_budget():
    """The widening rule earns its place — someone under the project's price is a lead for
    another project. Narrowing is_prospect must not have quietly taken that away."""
    prompt = _build_system_prompt(utc_now())
    assert "IS A VALID PROSPECT for other portfolio projects" in prompt


# --- whose name it is ---------------------------------------------------------------------


def test_the_agent_saying_the_name_is_not_the_prospect_saying_it():
    """The name is in the transcript because our own agent read it off the dial list."""
    assert name_spoken_by_prospect("Hari Shanker Choudhary", TRANSCRIPT) is False


def test_the_prospect_saying_their_name_counts():
    said = "Agent: May I know your name?\nProspect: Ravi Kumar."
    assert name_spoken_by_prospect("Ravi Kumar", said) is True


def test_a_short_name_is_still_checked():
    """phrase_is_grounded treats anything under four characters as unverifiable and lets it
    through. Here the fallback is better information, so absence of evidence is enough."""
    assert name_spoken_by_prospect("Ram", "Prospect: Hello.") is False
    assert name_spoken_by_prospect("Ram", "Prospect: My name is Ram.") is True


def test_the_dial_list_wins_when_the_prospect_never_said_a_name():
    """Otherwise the lead's one filled field is our own greeting read back to us."""
    assert resolve_customer_name("Hari Shanker Choudhary", False, "Hari Shanker Choudhary") == (
        "Hari Shanker Choudhary"
    )
    assert resolve_customer_name("Hari S Chodhury", False, "Hari Shanker Choudhary") == (
        "Hari Shanker Choudhary"
    )


def test_the_prospect_wins_when_they_said_it_themselves():
    """The person who answers is not always the person on the list, and a correction is the
    one case where the model knows more than the dial list does."""
    assert resolve_customer_name("Ravi", True, "Hari Shanker Choudhary") == "Ravi"


def test_with_no_contact_behind_the_call_the_extraction_is_all_there_is():
    """An inbound call, or one placed before the queue existed, has no dial list entry."""
    assert resolve_customer_name("Ravi", False, None) == "Ravi"


def test_nothing_anywhere_stays_nothing():
    assert resolve_customer_name(None, False, None) is None


def test_the_webhook_counts_prospect_words_and_not_the_transcript():
    """The helper being right is not enough — the bug was the call site using neither.

    Read off the keyword argument itself rather than grepping the function, so a mention of
    prospect_text anywhere else in the handler cannot make this pass. One level of
    indirection is followed — the count is now also used to log a cut-off call, so it is
    computed once into a local — but only one, and only to its assignment in this same
    function. `len(transcript.split())` still fails whichever way it is spelled.
    """
    import ast
    import inspect

    from app.api.routes import webhook

    tree = ast.parse(inspect.getsource(webhook._handle_call).lstrip())
    passed = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "answered_words"
    ]
    assert len(passed) == 1, [ast.unparse(p) for p in passed]

    expression = passed[0]
    if isinstance(expression, ast.Name):
        assignments = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id == expression.id
        ]
        assert len(assignments) == 1, f"{expression.id} is assigned more than once"
        expression = assignments[0]

    assert ast.unparse(expression) == "len(prospect_text(transcript).split())"
