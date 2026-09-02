"""Guards against the extractor recording the agent's own words as prospect data.

On a live call the prospect never named a budget — he asked what the 3 BHK cost and Priya
answered "priced at 1.8 Crores". The lead was stored with budget = 18000000, which is the
asking price read straight back out of the pitch. Budget is the field sales prioritise on.
"""

import ast
import inspect
from datetime import datetime

import pytest

from app import worker
from app.worker import _build_system_prompt

PROMPT = _build_system_prompt(datetime(2026, 7, 30, 9, 54, 0))


@pytest.mark.parametrize(
    "phrase,defect",
    [
        ("ASKING PRICE", "nothing tells the model the agent's figure is the price, not the budget"),
        ("never record either as the budget", "the prohibition itself is missing"),
        ("Asking what something costs is not having that budget", "the exact dodge that produced the bad row"),
        ("budget is null", "no instruction to leave budget empty when the prospect never gave one"),
        ("1.8 Crores", "the worked example is what makes the rule concrete"),
        ("Prospect:", "the rule must point at transcript turns the model can actually check"),
    ],
)
def test_budget_attribution_is_spelled_out(phrase, defect):
    assert phrase in PROMPT, defect


def test_location_attribution_survives():
    """The earlier 'Whitefield' defect — fixed once, must not regress while budget is rewritten."""
    assert "preferred_location" in PROMPT
    assert "Never copy one out of the Agent's pitch." in PROMPT


def test_transliteration_covers_names():
    """customer_name came back romanised while the transcript kept कुमार."""
    assert "Kumar" in PROMPT and "Priya" in PROMPT
    assert "No Devanagari character may remain" in PROMPT


def test_a_missing_status_is_never_stored_null():
    """13 of 24 leads had no status, including one with a 1.5 Crore budget and an agreed
    visit. A null hides the lead from every filter sales works from.

    The fill-in used to be a flat WARM. It is the lead's own ceiling now — never null, and
    never warmer than the call earned. See app/utils/lead_status.py."""
    from app.models.db import Lead, LeadStatus
    from app.utils.lead_status import capped_status

    assert capped_status(None, Lead()) is LeadStatus.COLD
    assert capped_status(None, Lead(budget=8_000_000)) is LeadStatus.WARM


def test_the_status_is_settled_before_the_hot_upgrade():
    """Order matters: settling it after the site-visit upgrade would overwrite HOT."""
    src = inspect.getsource(worker.process_extraction)
    assert src.index("capped_status(") < src.index("lead.site_visit_time is not None")


def test_the_column_itself_refuses_null():
    """Belt and braces — the application default cannot help a writer that bypasses it."""
    from app.models.db import Lead, LeadStatus

    col = Lead.__table__.c.status
    assert not col.nullable, "a raw insert could still store a null status"
    assert col.default.arg is LeadStatus.WARM
    assert col.server_default is not None


def test_untransliterated_transcript_is_flagged():
    """It failed silently: nothing in the log said the transcript was still in Devanagari."""
    tree = ast.parse(inspect.getsource(worker.process_extraction).lstrip())
    checked = {
        ast.unparse(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "isascii"
    }
    # customer_name is already guarded; the transcript is the one that failed silently, so
    # naming the exact expression keeps the other guard from covering for a missing one.
    assert "transcript_record.full_text.isascii" in checked, (
        f"no isascii check on the stored transcript, only on {checked or 'nothing'}"
    )
