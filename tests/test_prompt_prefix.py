"""How much of the prompt two different calls have in common, byte for byte.

The provider caches prompt prefixes across requests. A prefix is shared between two calls
only up to the first byte that differs — after that, everything is a fresh byte stream and
is billed, rate-limited and read as such. That makes the ORDER of the prompt an economic
fact, separate from anything the model understands.

Measured on 10 Sep 2026, before this file existed: a ~4,800-token prompt of which ~272
tokens were shared between two calls. The break was at line 10 — the prospect's name —
with the entire rulebook sitting behind it. Every turn of every call resent ~4,500 tokens
that were identical for every prospect in the country.

That budget is also why the agent sounds like a receptionist. The prompt's own docstring
says it: written as rules rather than prose because each word is paid for on every turn.
Rules produce rule-following. The headroom to sound like a person comes from here.
"""

from app.prompts.agent_prompts import get_system_prompt

CALL_A = get_system_prompt("Project Name: ALPHA\nLocation: Whitefield\nPrice: 1.2 Cr", "Rahul")
CALL_B = get_system_prompt("Project Name: BETA\nLocation: Sarjapur Road\nPrice: 85 L", "Sunita")
CALL_NO_NAME = get_system_prompt("Project Name: GAMMA\nLocation: Hebbal", None)


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# --- the number that matters ----------------------------------------------------------------


def test_two_different_calls_share_almost_all_of_the_prompt():
    """Nine tenths, measured in characters — the ratio is what the cache sees, and the
    token count scales with it. Before the reorder this was under six percent."""
    shared = _shared_prefix(CALL_A, CALL_B)
    assert shared / len(CALL_A) > 0.9, f"only {shared}/{len(CALL_A)} chars shared"


def test_a_call_with_no_name_breaks_inside_the_name_line_too():
    """The no-name branch is a different NAME line, not a different rulebook. Two named
    calls agree on "NAME: the lead list says this number belongs to " before the names
    diverge; the no-name call diverges a few dozen characters earlier, right after "NAME: ".
    Both breaks have to sit inside that one line — not a byte above it."""
    name_at = CALL_A.index("NAME:")
    name_end = CALL_A.index("\n", name_at)
    for other in (CALL_B, CALL_NO_NAME):
        n = _shared_prefix(CALL_A, other)
        assert name_at <= n < name_end, f"broke at {n}, NAME line spans {name_at}-{name_end}"


# --- where the break is, and that nothing per-call sits above it ----------------------------


def test_the_prefix_ends_at_the_name_line_and_not_before():
    """The first differing byte must be inside the NAME line. Anything per-call placed
    higher moves the break up and silently throws the rules below it out of the cache."""
    n = _shared_prefix(CALL_A, CALL_B)
    name_at = CALL_A.index("NAME:")
    assert n >= name_at, (
        f"prefix breaks at {n}, but NAME: is at {name_at} — something per-call sits above it:\n"
        f"{CALL_A[max(0, n - 60):n + 60]!r}"
    )


def test_every_rule_section_comes_before_the_name_line():
    """The rulebook is the static block. All of it has to be above the first per-call byte."""
    name_at = CALL_A.index("NAME:")
    for section in ("SIMPLE ENGLISH", "CALL FLOW", "OBJECTIONS:", "TOOL:"):
        assert CALL_A.index(section) < name_at, f"{section} sits below the name line"


def test_the_name_line_still_precedes_the_campaign_context():
    """Both are per-call. The name is the shorter and the one the flow refers to first, so it
    goes ahead of the context rather than being buried after a page of facts."""
    assert CALL_A.index("NAME:") < CALL_A.index("Campaign Context")


def test_the_name_rule_itself_is_unchanged_by_the_move():
    """Moving it is not rewriting it. The greeting, the no-name branch and the "prove who you
    are" wording all have tests of their own in test_call_script.py; this only checks the
    line survived the journey intact."""
    line = next(l for l in CALL_A.splitlines() if l.startswith("NAME:"))
    assert "Rahul" in line
    assert "not asking them to prove who they are" in line
    no_name = next(l for l in CALL_NO_NAME.splitlines() if l.startswith("NAME:"))
    assert "May I know your good name?" in no_name


# --- the docstring carries the rule, so the next edit does not undo it ----------------------


def test_the_prompt_docstring_explains_the_ordering():
    """A rule that lives only in a test gets "fixed" by whoever tidies the prompt next and
    wonders why the name is at the bottom."""
    doc = get_system_prompt.__doc__
    assert "ORDER MATTERS" in doc
    assert "Put nothing per-call above the TOOL section" in doc
