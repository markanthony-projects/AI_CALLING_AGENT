"""Who the agent says it is calling from.

On a live call on 4 Sep 2026 the agent opened with:

    "Hi, Good morning RAHUL. I am Priya calling you from Abhee Codename New Dimension."

That is the project. A person calls from the company building it and names the project when
they get to describing it — but the two were one column, so there was nowhere to put the
difference.

The field is optional and every existing project has it empty, so most of these tests are
about the fallback: a project nobody has filled this in for must sound exactly as it did
before the column existed.

Two links in the chain are the ones that break quietly, and both have a test here. The
column can exist, the dashboard can save it, and the call can still never see it — because
the value reaches the agent through a hand-built dict in discovery.py, and that dict is
cached in Redis for 24 hours, so entries written before this field existed have no key at
all.
"""

import inspect

import pytest

from app.services.agent import build_opening_line, caller_identity
from app.utils.context_builder import build_campaign_context

PROJECT = "Abhee Codename New Dimension"
DEVELOPER = "Abhee Ventures"


# --- who we are ------------------------------------------------------------------------


def test_the_developer_is_who_we_call_from():
    assert caller_identity(PROJECT, DEVELOPER) == DEVELOPER


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_without_one_it_is_the_project_exactly_as_before(missing):
    """Every project in the database has this empty today. A schema change must not alter
    what a live call says before anybody has typed the new value in."""
    assert caller_identity(PROJECT, missing) == PROJECT


def test_a_stray_space_around_the_typed_name_is_not_spoken():
    """It arrives from a text input on a form."""
    assert caller_identity(PROJECT, "  Abhee Ventures ") == DEVELOPER


def test_the_greeting_introduces_the_developer():
    line = build_opening_line(PROJECT, "Rahul", developer_name=DEVELOPER)
    assert f"from {DEVELOPER}." in line
    assert "Codename" not in line


def test_the_greeting_without_a_developer_is_unchanged():
    assert build_opening_line(PROJECT, "Rahul") == build_opening_line(
        PROJECT, "Rahul", developer_name=None
    )
    assert f"from {PROJECT}." in build_opening_line(PROJECT, "Rahul")


# --- the two links that break quietly ----------------------------------------------------


def test_the_value_is_carried_in_the_dict_the_call_actually_reads():
    """The agent never sees a Project row. It sees a dict assembled by hand in discovery.py
    and cached; a column left out of that dict is a column the call cannot see, with nothing
    failing anywhere to say so."""
    from app.services import discovery

    src = inspect.getsource(discovery)
    assert '"developer_name": project.developer_name,' in src


def test_a_cache_entry_written_before_this_field_existed_does_not_break_a_call():
    """That dict is stored in Redis for 24 hours. Entries already there have no such key,
    so reading it with [...] would raise KeyError inside the call handler and kill the call
    outright — for a greeting that would otherwise simply have said the project name."""
    from app.api.routes import webhook

    src = inspect.getsource(webhook)
    assert 'developer_name=project.get("developer_name")' in src
    assert 'developer_name=project["developer_name"]' not in src


def test_editing_a_project_does_not_wait_a_day_to_be_heard():
    """The same cache. Editing a project already clears it — an earlier bug had an edited
    price still being spoken on live calls the next day — and this field rides on that."""
    from app.api.routes import dashboard

    src = inspect.getsource(dashboard.update_project)
    assert "invalidate_project_everywhere" in src


# --- what the model is told ---------------------------------------------------------------


def test_the_model_is_told_who_it_works_for():
    context = build_campaign_context(
        {"name": PROJECT, "developer_name": DEVELOPER, "locality": "Varthur"}
    )
    assert DEVELOPER in context
    assert f"Project Name: {PROJECT}" in context


def test_a_project_with_no_developer_says_nothing_about_one():
    """An invented employer is worse than none, and the greeting has already fallen back to
    the project name on its own."""
    context = build_campaign_context({"name": PROJECT, "locality": "Varthur"})
    assert "Developer" not in context


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_developer_is_treated_as_absent_by_the_context_too(blank):
    """Otherwise the model is handed "Developer: " and asked to introduce itself with it."""
    context = build_campaign_context(
        {"name": PROJECT, "developer_name": blank, "locality": "Varthur"}
    )
    assert "Developer" not in context


def test_the_script_names_the_developer_in_the_greeting_and_the_project_in_the_pitch():
    """The split is the whole point. Naming the developer twice tells the prospect nothing
    new; naming the project in the greeting is what produced the original line."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    greeting, pitch = prompt.split("2. OPENING GATE", 1)
    assert "the Developer in the campaign context" in greeting
    assert "[project name]" not in greeting, "the greeting still names the project"
    assert "We are launching a new project in [location]." in pitch
    assert "It is called [project name], and it is" in pitch


def test_the_fallback_is_written_into_the_rule_the_model_reads():
    """Most projects have no developer recorded. A rule that only describes the happy path
    leaves the model to invent something for all of them."""
    from app.prompts.agent_prompts import get_system_prompt

    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert "or the project name if there is none" in prompt


# --- and the way it gets in there ----------------------------------------------------------


def test_it_can_be_set_when_a_project_is_created():
    from app.api.routes.dashboard import ProjectCreate

    assert ProjectCreate(name="X", city="C", locality="L").developer_name is None
    assert ProjectCreate(
        name="X", city="C", locality="L", developer_name=DEVELOPER
    ).developer_name == DEVELOPER


def test_it_can_be_set_on_a_project_that_already_exists():
    """Every project in the database predates the column."""
    from app.api.routes.dashboard import ProjectUpdate

    req = ProjectUpdate(developer_name=DEVELOPER)
    assert req.model_dump(exclude_unset=True) == {"developer_name": DEVELOPER}


def test_it_comes_back_out_so_the_form_can_show_what_was_saved():
    """Not the schema — the mapper every read goes through. A field the schema accepts and
    the mapper never fills is saved, invisible, and looks to the operator like it did not
    save at all."""
    from types import SimpleNamespace

    from app.api.routes.dashboard import _project_from_row

    row = SimpleNamespace(
        id="p1", name=PROJECT, developer_name=DEVELOPER, city="Bengaluru",
        locality="Varthur", min_price=None, max_price=None, possession_status=None,
        rera_id=None, amenities=[], usps=[], config_json=[], nearby_facilities=None,
        campaigns=0,
    )
    assert _project_from_row(row).developer_name == DEVELOPER


def test_creating_a_project_stores_the_developer_that_was_typed():
    """The handler, not the request model. ProjectCreate can carry the field perfectly and
    create_project still never pass it to the row it builds."""
    import asyncio

    from app.api.routes.dashboard import ProjectCreate, create_project

    stored = []

    class _Db:
        def add(self, obj):
            stored.append(obj)

        async def commit(self):
            pass

        async def refresh(self, obj):
            pass

    req = ProjectCreate(
        name=PROJECT, developer_name=DEVELOPER, city="Bengaluru", locality="Varthur"
    )
    summary = asyncio.run(create_project(req, db=_Db()))

    assert stored and stored[0].developer_name == DEVELOPER
    assert summary.developer_name == DEVELOPER, "saved, then dropped on the way back"


def test_creating_a_project_without_one_stores_nothing_rather_than_a_blank():
    import asyncio

    from app.api.routes.dashboard import ProjectCreate, create_project

    stored = []

    class _Db:
        def add(self, obj):
            stored.append(obj)

        async def commit(self):
            pass

        async def refresh(self, obj):
            pass

    req = ProjectCreate(name=PROJECT, city="Bengaluru", locality="Varthur")
    asyncio.run(create_project(req, db=_Db()))
    assert stored[0].developer_name is None


def test_the_column_exists_on_the_model_and_allows_null():
    from app.models.db import Project

    column = Project.__table__.c.developer_name
    assert column.nullable, "existing rows have no developer and must stay valid"


def test_the_migration_adds_it_nullable_and_backfills_nothing():
    """A backfill would have to guess an employer for every existing project."""
    import pathlib

    src = pathlib.Path("alembic/versions/d5b81f0c3a72_projects_developer_name.py").read_text(
        encoding="utf-8"
    )
    assert 'op.add_column("projects", sa.Column("developer_name", sa.String(), nullable=True))' in src
    assert "UPDATE" not in src.upper().replace("UPGRADE", "")
