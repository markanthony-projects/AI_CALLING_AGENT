"""The part of a name a person actually says out loud.

The greeting spoke whatever the lead list held, verbatim. On live calls that produced:

    "Hi, Good morning RAHUL."           <- shouted, because the CRM row was capitalised
    "Hi, Good morning mahantesha."      <- and this one was not
    "Hi, Good afternoon Abhijit Kumar Singh."

Nobody greets a stranger with their full legal name, and a voice engine reads case as
emphasis. All three are the same defect: a database field going straight to a speaker.

Two rules, and the second is the one worth stating. A salutation stays with the name it
belongs to, because "Dr. Rahul" is how that person is addressed and "Rahul" is a demotion.
And case is only corrected when the whole word is one case: "RAHUL" and "mahantesha" carry
no information in their capitals, but "DeSouza" and "McKenna" do, and title-casing those
would introduce a mistake while fixing one.
"""

import re
from typing import Optional

# Kept with the first name rather than dropped. Written without the full stop and matched
# without it too, so "Dr" and "Dr." are the same salutation.
_SALUTATIONS = {
    "mr": "Mr.",
    "mrs": "Mrs.",
    "ms": "Ms.",
    "miss": "Miss",
    "dr": "Dr.",
    "prof": "Prof.",
    "shri": "Shri",
    "sri": "Sri",
    "smt": "Smt.",
    "col": "Col.",
    "capt": "Capt.",
    "maj": "Maj.",
    "gen": "Gen.",
    "adv": "Adv.",
    "eng": "Eng.",
}

# An initial rather than a name. Indian lead lists carry "R Kumar" and "K. S. Sharma" as
# often as they carry a first name first, and "Good morning R." is worse than using the
# next word along.
_INITIAL = re.compile(r"^[A-Za-z]\.?$")

# Anything outside Latin letters, digits, spaces and the punctuation a name can hold.
# Sarvam breaks up mid-word on mixed script, so a Devanagari row in the lead list would
# garble the one line the caller has not yet decided to trust.
_SPEAKABLE = re.compile(r"^[A-Za-z][A-Za-z'\-. ]*$")


def _cased(word: str) -> str:
    """Fix a word that is shouting or whispering; leave one that is spelled deliberately."""
    letters = [c for c in word if c.isalpha()]
    if letters and (word.isupper() or word.islower()):
        return word.capitalize()
    return word


def spoken_name(raw: Optional[str]) -> str:
    """How to address this person, or "" when the list gives nothing usable.

    Empty is a real answer and the greeting already handles it: it simply leaves the name
    out and the agent asks in its first reply. A wrong or garbled name is worse than none,
    because it is the first thing the prospect hears and it tells them the call is
    automated before the agent has said anything else.
    """
    words = (raw or "").strip().split()
    if not words:
        return ""

    salutation = ""
    first = _SALUTATIONS.get(words[0].rstrip(".").lower())
    if first:
        salutation, words = first, words[1:]

    # Skip initials to reach a word somebody is called by. If the whole name is initials
    # there is nothing to say, so the greeting goes without one.
    while words and _INITIAL.match(words[0]):
        words = words[1:]
    if not words:
        return ""

    name = _cased(words[0])
    if not _SPEAKABLE.match(name):
        return ""
    return f"{salutation} {name}".strip()
