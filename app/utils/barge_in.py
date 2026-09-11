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


# Agreeing while somebody else is talking. On a Hindi call this is constant — "haan",
# "achha", "ji", "theek hai" are how a listener shows they are still there, and the English
# equivalents are mixed in freely. None of them are a request to speak.
#
# The awkward part is that they are the same words as an answer. "Haan" replies to "Kya aap
# Sunday free hain?" just as readily as it acknowledges a sentence in progress. What tells
# them apart is not the words but the moment: this check only runs while the agent is
# mid-sentence, and a question that has been asked is a question the agent has finished
# asking. An answer arrives into silence; an acknowledgement arrives into speech.
#
# Nothing is lost by being wrong here, and that is what makes it safe. These words used to
# be DISCARDED, not deferred — MinWordsUserTurnStartStrategy calls trigger_reset_aggregation()
# on anything under min_words while the bot speaks, so a lone "haan" was thrown away. Held,
# it survives and is answered the moment the sentence ends.
#
# "nahi" and "no" are deliberately absent. A refusal is never a backchannel, and it is the
# one thing a prospect most needs to be able to say over the top of a sales pitch.
_ACK = (
    r"(?:haan|haa|han|hm+|m+h?|ji|jee|achha|acha|accha|achcha|theek|thik|sahi|bilkul"
    r"|hai|ok|okay|okey|right|yeah|yes|yep|yup|sure|mhm|uh[-\s]?huh"
    r"|हाँ|हां|हम+|जी|अच्छा|ठीक|सही|बिल्कुल|है)"
)
_ONLY_ACKNOWLEDGEMENT = re.compile(rf"^\s*(?:{_ACK}[\s,.!?।]*)+$", re.I)


def takes_the_floor(text: Optional[str]) -> bool:
    """Whether these words should cut the agent off mid-sentence.

    False for two kinds of thing, and only while the agent is speaking, which is the only
    time this is asked: somebody checking whether the line is alive, and somebody agreeing
    along with a sentence still in progress. Neither is a request to speak, and neither is
    thrown away — both are answered as soon as the sentence ends.

    Everything else takes the floor, which is every answer a person actually gives: "3 BHK",
    "Sunday", "nahi", "ek minute ruko", and any sentence at all, including one that opens
    with an acknowledgement — "haan main Sunday free hoon" is an answer, not a "haan".
    """
    if not text or not text.strip():
        return False
    # Both patterns are anchored, so match() and search() are equivalent here and no test
    # can tell them apart — the ^ and $ are what do the work. The anchoring is the whole
    # safety property: "haan main Sunday free hoon" is an answer, and without it this would
    # swallow every sentence that opens with agreement.
    if _ONLY_ACKNOWLEDGEMENT.match(text):
        return False
    # match() rather than search() says the intent, though the ^ and $ in the pattern are
    # what actually do the work — the two are equivalent here and no test can tell them
    # apart. The anchors are the guarantee: "Hello can you tell me the price" is a sentence
    # and takes the floor like any other.
    return not _NEVER_TAKES_THE_FLOOR.match(text)
