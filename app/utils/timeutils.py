import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))

BUSINESS_HOURS = (10, 20)

_WEEKDAY_INDEX = {
    "MONDAY": 0,
    "TUESDAY": 1,
    "WEDNESDAY": 2,
    "THURSDAY": 3,
    "FRIDAY": 4,
    "SATURDAY": 5,
    "SUNDAY": 6,
}

_TIME_PATTERN = re.compile(
    r"^\s*(?P<hour>\d{1,2})(?:[:.](?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)?\s*$", re.I
)


def utc_now() -> datetime:
    """Current UTC as a naive datetime, matching the TIMESTAMP WITHOUT TIME ZONE columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_ist(value: datetime) -> datetime:
    """Reinterpret a naive-UTC column value as IST.

    Call timestamps are stored in UTC, but everything the model reasons about
    ("tomorrow at 11", "this evening") and everything sales acts on is local time.
    Handing the model a UTC clock labelled IST shifted every relative date by 5h30m.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(IST)


def time_of_day_greeting(value: Optional[datetime] = None) -> str:
    """"morning" / "afternoon" / "evening" for the prospect's clock, not the server's.

    Read in IST because the droplet runs on UTC: at 09:30 IST — the middle of a dialing
    shift — a UTC clock says 04:00, and the caller would be wished good morning at what the
    machine thinks is the dead of night. The 5h30m offset also moves the afternoon boundary
    by half a day, so getting this from utcnow() is wrong twice over.

    Boundaries follow Indian English usage: afternoon starts at noon, evening at 5 PM.
    Anything before noon is morning — dialing runs 10 AM to 8 PM, so the small hours never
    come up, and "good morning" is the harmless answer if they ever do.
    """
    hour = to_ist(value if value is not None else utc_now()).hour
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def parse_time_of_day(raw: Optional[str]) -> Optional[time]:
    """Parse '15:00', '3 PM', '3:30pm' into a time. None if it isn't a usable clock time."""
    if not raw:
        return None
    match = _TIME_PATTERN.match(str(raw))
    if not match:
        return None

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    meridiem = (match.group("meridiem") or "").lower()

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return time(hour=hour, minute=minute)


def _next_occurrence(reference: datetime, weekday_index: int, at: time) -> date:
    """The soonest date on/after reference that falls on weekday_index and is still ahead."""
    day = reference.date() + timedelta(days=(weekday_index - reference.weekday()) % 7)
    if datetime.combine(day, at) <= reference:
        day += timedelta(days=7)
    return day


def resolve_appointment(
    reference: datetime,
    *,
    weekday: Optional[str] = None,
    in_days: Optional[int] = None,
    time_of_day: Optional[str] = None,
) -> Optional[datetime]:
    """Resolve an extracted appointment intent into a concrete local datetime.

    Returns None unless BOTH a day reference and a time were actually agreed. A partial
    commitment is not an appointment: defaulting the missing half is what put a Thursday
    11:00 site visit in the database for a prospect who only ever said "yeah sure".
    """
    at = parse_time_of_day(time_of_day)
    if at is None:
        return None

    ref = reference.replace(tzinfo=None) if reference.tzinfo else reference

    if in_days is not None:
        day = (ref + timedelta(days=in_days)).date()
    elif weekday is not None:
        index = _WEEKDAY_INDEX.get(str(getattr(weekday, "value", weekday)).upper())
        if index is None:
            return None
        day = _next_occurrence(ref, index, at)
    else:
        return None

    return datetime.combine(day, at)


def is_within_business_hours(value: datetime) -> bool:
    start, end = BUSINESS_HOURS
    return start <= value.hour < end
