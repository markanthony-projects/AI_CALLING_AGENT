"""Turning what a lead list contains into something a carrier will dial.

Lifted out of app/api/routes/campaign.py, where it lived while dialling was only ever
triggered by an HTTP request carrying a list of numbers. The spreadsheet importer and the
dial pump both need it now, and a service reaching into a routes module for its phone
parsing is the wrong direction — the route is one caller of this, not its owner.
"""

import re

from app.core.config import settings

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")
# Separators only. A dot is deliberately not one: "9876543210.0" is a number Excel has
# turned into a float, and stripping the dot would make it an 11-digit number that the
# carrier dials, bills, and fails. It is handled as a float artefact below instead.
_SEPARATORS = re.compile(r"[\s\-()]")
# "9876543210.0" — what a spreadsheet does to a phone column stored as a number.
_FLOAT_TAIL = re.compile(r"\.0+$")

# How many digits a national number has, per country code. E.164 alone allows 8 to 15 and
# would pass "+91987654321" — nine digits, a lead-list typo — straight to the carrier,
# which dials it, bills it and fails. Every Indian subscriber number, mobile or landline
# with its STD code, is exactly ten. Only countries listed here are length-checked.
_NATIONAL_DIGITS = {"91": 10}


def to_e164(raw: str) -> str:
    """Accept the formats a lead list actually contains and return strict E.164.

    Spreadsheets and CRM exports carry numbers as '98765 43210', '098765-43210',
    '91 9876543210' or '9876543210.0'; dialing must not fail on punctuation the operator
    never chose, and must not dial a number the spreadsheet mangled.

    Raises ValueError on anything that cannot be dialled. Callers differ in what they do with
    that: an API request should refuse, while an import marks the row INVALID and carries on
    with the rest of the file.
    """
    cc = settings.DEFAULT_COUNTRY_CODE
    cleaned = _FLOAT_TAIL.sub("", _SEPARATORS.sub("", str(raw).strip()))

    if cleaned.startswith("+"):
        candidate = cleaned
    else:
        digits = cleaned.lstrip("0")  # national trunk prefix, or 00 international prefix
        # A bare national number is 10 digits; only a longer one can already carry
        # the country code. Guards against 10-digit mobiles that start with 91.
        if len(digits) > 10 and digits.startswith(cc):
            candidate = f"+{digits}"
        else:
            candidate = f"+{cc}{digits}"

    if not _E164.match(candidate):
        raise ValueError(
            f"'{raw}' is not a dialable number. Use E.164 (+919876543210) "
            f"or a local number that a +{cc} prefix completes."
        )

    for code, length in _NATIONAL_DIGITS.items():
        if candidate.startswith(f"+{code}") and len(candidate) - 1 - len(code) != length:
            raise ValueError(
                f"'{raw}' is not a dialable +{code} number: it has "
                f"{len(candidate) - 1 - len(code)} digits after the country code and "
                f"needs {length}."
            )
    return candidate
