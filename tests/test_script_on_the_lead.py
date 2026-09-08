"""Which alphabet the lead sheet gets written in.

Live call 7 Sep 2026. The prospect asked for Sarjapur Road and the lead recorded:

    preferred_location = 'सरजापुर road'

Nothing here was invented. The prospect really said it, the model really transliterated the
transcript as instructed, and it really did leave the extracted field in Devanagari — because
only customer_name's description ever told it not to. Sales opened a lead they could not read.

The second half is worse and was found while fixing the first. app/utils/attribution.py exists
to answer "is there a Prospect line this value could have come from", and it looks for
distinctive words with `[a-z0-9]+`. A Devanagari value yields no tokens at all, so it lands in
the deliberate "nothing long enough to check" branch — the one that keeps '2 BHK' — and is
returned as grounded without having been compared to anything. A locality the model invented
in Devanagari reaches the CRM by exactly the same door a real one does.

So script is checked separately, and first.
"""

import ast
import inspect
import typing

import pytest

from app.models.schemas import LeadExtraction
from app.utils.attribution import is_readable, phrase_is_grounded

# The value from the call, and the transcript it came off.
LIVE_VALUE = "सरजापुर road"  # सरजापुर road
TRANSCRIPT = "Agent: We are launching in Varthur - Sarjapur Road\nProspect: haan theek hai"


# --- reading the script -----------------------------------------------------------------


def test_the_value_from_the_live_call_is_not_readable():
    assert is_readable(LIVE_VALUE) is False


def test_the_same_place_in_latin_letters_is():
    """The fix must not reject the answer, only the alphabet."""
    assert is_readable("Sarjapur Road") is True


# Each sample is written with explicit codepoints and carries the Unicode block it is
# supposed to exercise, because eyeballing this got it wrong once — see the purity test
# below.
SCRIPT_SAMPLES = [
    ("Devanagari", "कोरमंगला", (0x0900, 0x097F)),
    ("Bengali", "কলকাতা", (0x0980, 0x09FF)),
    ("Gurmukhi", "ਲੁਧਿਆਣਾ", (0x0A00, 0x0A7F)),
    ("Gujarati", "અમદાવાદ", (0x0A80, 0x0AFF)),
    ("Tamil", "சென்னை", (0x0B80, 0x0BFF)),
    ("Telugu", "హైదరాబాద్", (0x0C00, 0x0C7F)),
    ("Kannada", "ಕೋರಮಂಗಲ", (0x0C80, 0x0CFF)),
    ("Malayalam", "കൊച്ചി", (0x0D00, 0x0D7F)),
]


@pytest.mark.parametrize("script,value,block", SCRIPT_SAMPLES)
def test_every_script_the_stt_can_return_is_caught(script, value, block):
    """STT_LANGUAGE can name any of these. Catching only Hindi would leave the same bug
    waiting for the first Kannada campaign."""
    assert is_readable(value) is False, script


@pytest.mark.parametrize("script,value,block", SCRIPT_SAMPLES)
def test_each_sample_is_written_only_in_the_script_it_names(script, value, block):
    """A sample that borrows one letter from a neighbouring script silently tests that
    neighbour instead, and the range it was meant to cover can then be deleted with every
    test still green.

    That is not hypothetical. The Kannada sample here was first written by hand as
    ಕೋರಮಂగಲ, whose sixth letter is a Telugu ga — so it was matched by the Telugu range, and
    removing Kannada from _INDIC broke nothing. Mutation testing found it; reading it did
    not, because the two letters look almost identical at this size."""
    low, high = block
    stray = [hex(ord(c)) for c in value if not low <= ord(c) <= high]
    assert not stray, f"{script} sample contains characters from another block: {stray}"


@pytest.mark.parametrize("value", [None, "", "2 BHK", "Sarjapur Road", "South Bengaluru"])
def test_nothing_ordinary_is_dropped(value):
    assert is_readable(value) is True


def test_a_latin_letter_with_an_accent_is_still_readable():
    """The test is "did the model skip transliteration", not "is this ASCII". A lead sheet
    can hold an accent; it cannot hold an alphabet its reader does not have."""
    assert is_readable("Café Road") is True


# --- the hole this closes ---------------------------------------------------------------


def test_attribution_passes_a_devanagari_value_it_never_checked():
    """Not a bug in phrase_is_grounded — it is doing what its docstring says. It is the
    reason script cannot be left to it: on this transcript the prospect said nothing but
    "haan theek hai", and an invented Devanagari locality is returned as grounded."""
    invented = "कोरमंगला"  # कोरमंगला
    assert phrase_is_grounded(invented, TRANSCRIPT) is True
    assert is_readable(invented) is False


def test_the_live_value_slips_through_attribution_too():
    """"road" is the only token in it, and _GENERIC throws that away as worthless proof —
    so this reaches the same unchecked branch."""
    assert phrase_is_grounded(LIVE_VALUE, TRANSCRIPT) is True


# --- what the worker does with it -------------------------------------------------------


def _worker_source(name: str) -> str:
    from app import worker

    return inspect.getsource(getattr(worker, name))


def _lead(**fields) -> LeadExtraction:
    return LeadExtraction(is_prospect=True, **fields)


def _drop(lead):
    from app.worker import _drop_unreadable

    return _drop_unreadable(lead, "sid")


def test_the_unreadable_location_is_nulled():
    assert _drop(_lead(preferred_location=LIVE_VALUE)).preferred_location is None


def test_a_readable_location_is_left_alone():
    kept = _drop(_lead(preferred_location="Sarjapur Road"))
    assert kept.preferred_location == "Sarjapur Road"


def test_only_the_unreadable_field_is_lost():
    """A Devanagari locality must not take the budget down with it — dropping more than the
    one bad field is how a good call ends up with an empty lead."""
    out = _drop(_lead(preferred_location=LIVE_VALUE, budget=15000000.0, timeline="2 months"))
    assert out.preferred_location is None
    assert out.budget == 15000000.0
    assert out.timeline == "2 months"


def test_a_lead_with_nothing_wrong_is_returned_unchanged():
    lead = _lead(customer_name="Rahul", preferred_location="Whitefield")
    assert _drop(lead) is lead


# --- and it covers every field that can carry a script ----------------------------------

# Free text is anything a person's words land in. These three are Optional[str] and are not:
# the two times are validated as "HH:MM", and the transliterated transcript is the thing the
# romanised values are recovered from — process_extraction warns about that one separately,
# and nulling it would throw away the only record of what was actually said.
NOT_FREE_TEXT = {"site_visit_at", "callback_at", "transliterated_transcript"}


def _optional_str_fields() -> set[str]:
    found = set()
    for name, field in LeadExtraction.model_fields.items():
        annotation = field.annotation
        args = typing.get_args(annotation)
        if annotation is str or (str in args and type(None) in args):
            found.add(name)
    return found


def test_every_free_text_field_is_checked():
    """Parsed off the schema rather than listed here, so a free-text field added later
    fails this test instead of quietly reaching the CRM in Devanagari."""
    from app.worker import _WRITTEN_FIELDS

    assert _optional_str_fields() - NOT_FREE_TEXT == set(_WRITTEN_FIELDS)


def test_the_fields_deliberately_left_out_are_still_the_ones_named_here():
    """If one of these stops existing the exemption is stale, and a stale exemption is how
    a field goes unchecked without anyone deciding that it should."""
    assert NOT_FREE_TEXT <= _optional_str_fields()


# --- order ------------------------------------------------------------------------------


def test_script_is_checked_before_attribution():
    """Attribution cannot read the value, so it passes it. Running it first would report a
    Devanagari hallucination as grounded and then drop it for the wrong reason — and if the
    drop were ever removed, silently keep it."""
    from app import worker

    src = inspect.getsource(worker.process_extraction)
    assert src.index("_drop_unreadable(lead_data") < src.index("_drop_ungrounded(lead_data")


def test_the_script_check_does_not_need_the_transcript():
    """It asks what alphabet the value is in, which the transcript cannot change. Taking one
    would invite it to be checked against the transliterated text and pass everything."""
    tree = ast.parse(_worker_source("_drop_unreadable").lstrip())
    args = [a.arg for a in tree.body[0].args.args]
    assert args == ["lead_data", "call_sid"]


def test_the_call_that_produced_the_rule_is_written_beside_it():
    doc = _worker_source("_drop_unreadable")
    assert "सरजापुर" in doc
    assert "7 Sep 2026" in doc


# --- the prompt half --------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["customer_name", "preferred_location", "preferred_unit_type", "timeline"]
)
def test_the_model_is_told_which_script_to_use(field):
    """customer_name had this and came back romanised on every call. preferred_location did
    not, and did not. The instruction works — it was simply never given."""
    description = LeadExtraction.model_fields[field].description
    assert "Latin script" in description or "Latin" in description
    assert "Devanagari" in description


def test_the_location_carries_a_worked_example():
    """The two rules this file's neighbour enforces were both ignored as bare instructions
    and obeyed once an example sat under them."""
    description = LeadExtraction.model_fields["preferred_location"].description
    assert "Sarjapur" in description


def test_the_code_is_a_backstop_and_not_the_only_rule():
    """Dropping the field loses a real preference. The prompt is what keeps it, and the code
    is what makes the loss visible when the prompt fails."""
    from app.worker import _drop_unreadable

    assert "prompt" in inspect.getdoc(_drop_unreadable)
