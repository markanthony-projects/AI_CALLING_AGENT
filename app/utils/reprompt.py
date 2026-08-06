"""Rescue the call when the prospect answers and the transcript comes back empty.

Twice in one live call the pipeline went mute on the prospect. Both times the shape was
identical:

    AGENT → "Got it, Shivam. Is this for your own stay, or for investment?"
    VAD fired with no transcribable speech after 5003ms
    USER  → "Hello."                                        <- 11 seconds later
    AGENT → "I am sorry, Shivam. I think the line lagged. Were you looking for..."
    USER  → "Yeah, I SAID for investment."

The prospect had answered. Deepgram returned nothing for five seconds of audio, so no
inference ran, so the agent said nothing at all — and the only thing that restarted the
conversation was the prospect giving up and saying "Hello?" into the silence. Between the
two of them that cost about twenty seconds of a hundred-and-eighty-second call, and it made
us look like the broken party.

Why the transcript was empty is a separate question and SttWitness exists to answer it on
the next occurrence. This module handles the part that is worth fixing either way: an agent
that has asked a question and heard nothing back should ask it again, not wait to be
rescued. The model already produces exactly the right recovery when it is finally given a
turn ("I think the line lagged" plus the question again) — this simply does it without
needing the prospect's permission first, and without an LLM round trip.

Only ever a question is repeated. If the last thing the agent said was a sign-off there is
nothing to re-ask and re-speaking it would be a second goodbye.
"""

import re
from typing import Optional

# Spoken in front of the repeated question. Deliberately blames the line rather than the
# prospect: they answered, we did not hear it, and "sorry, I missed that" reads as though
# they mumbled. Kept to one short clause because the question behind it is the point.
DEAD_AIR_APOLOGY = "Sorry, I think the line dropped for a moment."

# A line that swallows one answer is bad luck; a line that swallows three is a line that is
# not going to carry this call, and repeating ourselves into it just wastes the prospect's
# patience. After this the idle timeout takes over.
MAX_DEAD_AIR_NUDGES = 2

# Sentence-final punctuation, keeping the mark with the sentence it ends.
_SENTENCE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")


def last_question(agent_line: Optional[str]) -> Optional[str]:
    """The question at the end of the agent's last turn, or None if it did not ask one.

    The last one rather than the first: a reply is an acknowledgement plus a question, and
    on the rare turn carrying two it is the closing one the prospect was answering.
    """
    if not agent_line:
        return None
    for sentence in reversed([s.strip() for s in _SENTENCE.findall(agent_line)]):
        if sentence.endswith("?"):
            return sentence
    return None


def dead_air_nudge(agent_line: Optional[str]) -> Optional[str]:
    """What to say when the prospect's turn produced no transcript, or None to stay quiet."""
    question = last_question(agent_line)
    if not question:
        return None
    return f"{DEAD_AIR_APOLOGY} {question}"
