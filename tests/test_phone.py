"""What the dialer will and will not ring.

E.164 alone allows eight to fifteen digits, so the first version passed "+91987654321" —
nine digits, a lead-list typo — and "+9198765432100" — a spreadsheet's float, ".0" and
all — straight to the carrier, which dialled both, billed both, and connected neither.
Every Indian subscriber number is exactly ten digits after +91, mobile or landline with
its STD code, and a number that is not ten digits is not a number to pay to ring.
"""

import pytest

from app.utils.phone import to_e164


@pytest.mark.parametrize(
    "raw",
    [
        "9876543210",
        "09876543210",
        "+91 98765 43210",
        "919876543210",
        "0091 9876543210",
        "91 98765-43210",
        "(0)98765 43210",
        "9876543210.0",  # Excel turned the column into floats
        "919876543210.00",
    ],
)
def test_every_spelling_of_the_same_number_dials_the_same_number(raw):
    assert to_e164(raw) == "+919876543210"


@pytest.mark.parametrize(
    "raw",
    [
        "98765 4321",  # nine digits
        "98765432101",  # eleven
        "+91987654321",
        "+9198765432101",
        "9.87654321E9",  # a float nobody can undo
        "",
        "call me",
    ],
)
def test_a_number_that_is_not_ten_digits_is_refused_before_the_carrier_sees_it(raw):
    with pytest.raises(ValueError):
        to_e164(raw)


def test_the_refusal_says_how_many_digits_it_found():
    with pytest.raises(ValueError, match="9 digits after the country code"):
        to_e164("98765 4321")


def test_another_country_is_still_checked_only_by_e164():
    """The ten-digit rule is +91's. A number for a country this does not know keeps the
    E.164 check alone, rather than being refused by a rule written for somewhere else."""
    assert to_e164("+14155552671") == "+14155552671"
