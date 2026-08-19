"""Turning what a lead list contains into something a carrier will dial.

Lifted out of app/api/routes/campaign.py, where it lived while dialling was only ever
triggered by an HTTP request carrying a list of numbers. The spreadsheet importer and the
dial pump both need it now, and a service reaching into a routes module for its phone
parsing is the wrong direction — the route is one caller of this, not its owner.
"""

import re

from app.core.config import settings

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")
_SEPARATORS = re.compile(r"[\s\-().]")


def to_e164(raw: str) -> str:
    """Accept the formats a lead list actually contains and return strict E.164.

    Spreadsheets and CRM exports carry numbers as '98765 43210', '098765-43210' or
    '91 9876543210'; dialing must not fail on punctuation the operator never chose.

    Raises ValueError on anything that cannot be dialled. Callers differ in what they do with
    that: an API request should refuse, while an import marks the row INVALID and carries on
    with the rest of the file.
    """
    cc = settings.DEFAULT_COUNTRY_CODE
    cleaned = _SEPARATORS.sub("", str(raw).strip())

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
    return candidate
