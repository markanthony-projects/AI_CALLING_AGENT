"""Which words are allowed to cut the agent off mid-sentence.

Live call 5023ff25, 10 Sep 2026. The agent went mute for the last thirty seconds and the
prospect hung up. The shape:

    AGENT -> "...Which size are you thinking of?"
    USER  -> "Hello."                          TTS reconnected (3)
    AGENT -> "Hi, could you tell me..."        [interrupted]
    USER  -> "Hello."                          TTS reconnected (4)
    AGENT -> "Which BHK size..."               [interrupted]
    (prospect hangs up)

"Hello" is one word. The barge-in gate relaxes to one word after the greeting, so every one
of them took the floor — and Sarvam's TTS is an InterruptibleTTSService, which tears down
and reopens its websocket on every interruption. Four reconnects, and the audio stopped
arriving. The prospect was saying "Hello?" BECAUSE they could not hear, and each "Hello?"
was what stopped them hearing.

The gate was set to one word for a real reason, recorded on GreetingOnlyMinWords: at three,
Pipecat's MinWordsUserTurnStartStrategy DISCARDS anything shorter while the bot is speaking
— `trigger_reset_aggregation()` — so "Yeah sure." answering "Would you like to visit?"
vanished and a caller sat through 37 seconds of silence. One word was a workaround for the
discarding, not a decision that every word should interrupt.

So the rule is about the words rather than the count. "Hello?" while the agent is talking is
not someone taking the floor; it is someone who cannot hear, and the worst possible response
is to stop talking. It does not interrupt — and it is not thrown away either. It is held,
and the turn starts the moment the agent finishes its sentence, which is what a person would
do: finish the thought, then answer.
"""

import re
from typing import Optional

# Checking the line, or getting attention. Never an answer to anything, so nothing is lost
# by letting the agent finish first. Anchored whole-string: "Hello, yes, who is this?" is a
# conversation and takes the floor like any other sentence.
_NEVER_TAKES_THE_FLOOR = re.compile(
    r"^\s*(?:"
    r"hello|hallo|helo"
    r"|sir|madam|ma'?am"
    r"|are you there|is anyone there|can you hear me|are you still there"
    r"|hello\s*\?*\s*hello"
    r")\s*[?.!,]*\s*$",
    re.I,
)


def takes_the_floor(text: Optional[str]) -> bool:
    """Whether these words should cut the agent off mid-sentence.

    False only for the short line-checks above. Everything else — including every one- and
    two-word answer a person actually gives, "Yeah", "3 BHK", "Sunday" — interrupts exactly
    as it does today.
    """
    if not text or not text.strip():
        return False
    # match() rather than search() says the intent, though the ^ and $ in the pattern are
    # what actually do the work — the two are equivalent here and no test can tell them
    # apart. The anchors are the guarantee: "Hello can you tell me the price" is a sentence
    # and takes the floor like any other.
    return not _NEVER_TAKES_THE_FLOOR.match(text)
