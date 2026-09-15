"""The first thing the caller hears, built where the worker can build it too.

Moved out of app/services/agent.py on 15 Sep 2026 so the opening line can be assembled at
dial time, in the worker, and its sentences synthesised while the phone is still ringing —
the worker is kept free of the pipecat runtime on purpose (see app/services/dialer.py), and
agent.py imports all of it. Nothing here needs a pipeline: a name, a company, the time of
day, and the sentence that hands the prospect the turn.

The three builders are unchanged from where they were; agent.py re-exports them, so every
caller and every test that imports them from there keeps working.
"""

from datetime import datetime
from typing import Optional

from app.prompts.agent_prompts import AGENT_NAME, introduction
from app.utils.person_name import spoken_name
from app.utils.timeutils import time_of_day_greeting


def caller_identity(project_name: str, developer_name: Optional[str] = None) -> str:
    """Who the agent says it is calling from.

    The developer, when the project has one recorded. On a live call the agent said "I am
    Priya calling you from Abhee Codename New Dimension" — which is the project, not an
    employer. A person calls from the company and names the project when they describe it.

    Falls back to the project name, which is what every call has said until now, so a
    project nobody has filled this in for sounds exactly as it did before.
    """
    return (developer_name or "").strip() or project_name


def build_opening_line(
    project_name: str,
    customer_name: Optional[str] = None,
    now: Optional[datetime] = None,
    developer_name: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> str:
    """The first thing the caller hears.

    Greets by time of day and says who is calling, and nothing else. The name off the dial
    list is used to address them, so a prospect who does hear their own name knows the call
    is meant for them; without one the greeting simply omits it and the agent asks in its
    first reply — never a guessed name.

    It ends by asking for a minute of their time, and that was removed once and put back.

    The argument for removing it was that eight words carried no information and invited a
    "no" to a question that was not the one worth asking, since the opening gate asks for
    the same permission and sorts the call as well. Half of that was right and the half that
    was wrong cost more: nothing replaced them. A greeting that ends on a statement hands
    the prospect nothing to answer, and on a live call on 5 Sep the line went quiet for
    three seconds and then they had to ask "What is the purpose?" themselves.

    A question is what passes the turn over. The words are not there for their information;
    they are there so the other person knows it is their go.

    Written as short sentences rather than one comma-spliced line, and SPOKEN as short
    sentences too — see spoken() below. Pipecat synthesises the model's replies one sentence
    per request, so there a full stop is a real gap the caller hears while a comma is not:
    measured on bulbul:v3, the same words with and without commas take the same time to say.
    But a TTSSpeakFrame skips that per-sentence cut, and this line was going to the voice
    engine as one request, three sentences in one flat breath. On 10 Sep 2026 it was the
    one line on the call reported as sounding like a machine. The full stops only became
    real gaps once the line was queued one sentence per frame.
    """
    part = time_of_day_greeting(now)
    # The lead list holds "Abhijit Kumar Singh", "RAHUL" and "mahantesha"; none of those is
    # how a person is greeted. Idempotent, so applying it here as well as before the prompt
    # costs nothing and means this line is safe whoever calls it. See app/utils/person_name.
    name = spoken_name(customer_name)
    # "Prestige Pvt. Ltd." already ends in a full stop; the sentence supplies its own.
    identity = caller_identity(project_name, developer_name).rstrip(".")
    who = (agent_name or "").strip() or AGENT_NAME
    # Ends on a question about THEM, not on permission. "Can I speak to you for a minute?"
    # invites a no from somebody who has not heard anything yet; "Am I speaking with Rahul?"
    # invites a yes, and confirms we reached the person the dial list named. Without a name
    # it asks for one, which is the same question pointed the other way.
    ask = f"Am I speaking with {name}?" if name else "May I know your good name?"
    # The introduction carries the AI disclosure and is built in one place, shared with the
    # prompt, so the system-spoken greeting and the model's own introduction cannot drift.
    return f"Hello, Good {part}. {introduction(who, identity)} {ask}"


def build_reintroduction(
    project_name: str,
    customer_name: Optional[str] = None,
    developer_name: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> str:
    """Said when their first words are "Hello?" — they heard the line, not the sentence.

    The greeting without the time of day. "Good afternoon" is true once; said twice inside
    ten seconds it is the single most obviously automated thing a caller can hear. What has
    to come back is the part they missed: who this is, and the question that hands them the
    turn.
    """
    name = spoken_name(customer_name)
    identity = caller_identity(project_name, developer_name).rstrip(".")
    who = (agent_name or "").strip() or AGENT_NAME
    ask = f"Am I speaking with {name}?" if name else "May I know your good name?"
    return f"{introduction(who, identity)} {ask}"
