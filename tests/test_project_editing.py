"""Editing the facts the agent speaks.

Project management existed — create, edit, delete, all wired to the dashboard — but two
fields were missing from it, and they were the two the caller actually hears. A project
created through the dashboard had no configurations and no location facts, so the agent
could not say which homes the project sells or what is near it. Both read as the agent
being vague; neither was fixable from anywhere in the UI.

The design rests on one claim, and it is the claim these pin: flattening a location fact to
text loses nothing, because flattening is the last thing that happens to it before the model
sees it. If that stops being true, an operator reads one thing and the caller hears another.
"""

import pytest
from pydantic import ValidationError

from app.api.routes.dashboard import ProjectCreate, ProjectUpdate, _nearby_from_column
from app.utils.context_builder import build_campaign_context, spoken_facility


# --- flattening is lossless where it counts -------------------------------------------


def test_a_structured_facility_and_its_flattened_form_reach_the_model_identically():
    """The whole reason the dashboard is allowed to store text. spoken_facility runs on the
    way to the prompt, so a value stored already-flattened produces the same line."""
    structured = [{"name": "Whitefield Metro", "drive_time": "8 min", "distance": "3 km"}]
    flattened = spoken_facility(structured)
    assert spoken_facility(flattened) == flattened


def test_the_whole_prompt_is_unchanged_by_flattening():
    """Asserted on the built prompt and not only on the helper: a caller that formats the
    value some other way would break the round trip without touching spoken_facility."""
    structured = {"Metro": [{"name": "Whitefield", "drive_time": "8 min"}], "School": "Ryan, 2 km"}
    project = {"name": "P", "city": "Bengaluru", "locality": "Whitefield"}

    rich = build_campaign_context({**project, "nearby_facilities": structured})
    flat = build_campaign_context({**project, "nearby_facilities": _nearby_from_column(structured)})
    assert rich == flat


def test_a_list_of_strings_flattens_and_stays_flat():
    assert spoken_facility(["Metro 3 km", "Mall 1 km"]) == "Metro 3 km, Mall 1 km"


# --- what the operator is shown ---------------------------------------------------------


def test_facilities_are_shown_as_the_agent_would_say_them():
    nearby = _nearby_from_column({"Metro": {"name": "Whitefield", "drive_time": "8 min"}})
    assert nearby == {"Metro": "Whitefield, 8 min"}


def test_a_column_the_agent_cannot_read_is_shown_as_empty():
    """context_builder ignores anything that is not a dict of categories, so the agent has
    no location facts at all. Rendering the raw shape would show the operator facts the
    caller will never hear, and hide that the project needs filling in."""
    assert _nearby_from_column([{"name": "Metro"}]) == {}
    assert _nearby_from_column(None) == {}
    assert _nearby_from_column("Metro 3 km") == {}


def test_a_category_with_nothing_in_it_is_not_shown():
    assert _nearby_from_column({"Metro": "", "School": [], "Mall": "Phoenix"}) == {"Mall": "Phoenix"}


# --- configurations are validated at the boundary --------------------------------------


def test_configurations_accept_what_the_agent_reads():
    body = ProjectCreate(
        name="P", city="Bengaluru", locality="Whitefield",
        config_json=[{"type": "3 BHK", "area": "1450 sqft", "price": "1.17 Cr"}],
    )
    assert body.config_json[0].type == "3 BHK"
    assert body.config_json[0].price == "1.17 Cr"


def test_a_configuration_that_is_not_an_object_is_refused():
    """The agent reads this list out loud. A bare string here used to be stored happily and
    then silently skipped by the isinstance guard in context_builder, so the project simply
    stopped mentioning that unit — with nothing anywhere to say why."""
    with pytest.raises(ValidationError):
        ProjectCreate(name="P", city="C", locality="L", config_json=["3 BHK"])


def test_configurations_may_be_left_out_entirely():
    """Distinct from an empty list: unset means 'do not touch', which is what a PATCH of the
    name alone has to mean."""
    assert "config_json" not in ProjectUpdate(name="P").model_dump(exclude_unset=True)


def test_a_patch_sends_plain_dicts_to_the_jsonb_column():
    """setattr puts this straight on the ORM object, which cannot serialise a pydantic model."""
    dumped = ProjectUpdate(
        config_json=[{"type": "2 BHK", "area": "1100 sqft", "price": "92 L"}]
    ).model_dump(exclude_unset=True)
    assert dumped["config_json"] == [{"type": "2 BHK", "area": "1100 sqft", "price": "92 L"}]


def test_facilities_can_be_cleared_but_not_by_accident():
    """An explicit empty dict wipes the column; an unset field leaves it alone. The agent
    loses every location fact in the first case, so the two must not be the same request."""
    assert ProjectUpdate(nearby_facilities={}).model_dump(exclude_unset=True) == {
        "nearby_facilities": {}
    }
    assert "nearby_facilities" not in ProjectUpdate(city="Pune").model_dump(exclude_unset=True)


# --- the edit actually reaching the caller ----------------------------------------------
#
# Saving a project is only half of it. discovery.py caches the project context per campaign —
# five minutes in-process, a day in Redis — and both project routes invalidated it with the
# project id. The cache has no key of that shape, so editing a price cleared nothing and the
# agent kept quoting the old one for the rest of the day. The dashboard reported success
# either way, which is why nobody would find this from the outside.


class FakeRedis:
    def __init__(self):
        self.deleted = []

    async def delete(self, key):
        self.deleted.append(key)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class FakeDb:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _stmt):
        return FakeResult(self._rows)


@pytest.fixture
def redis(monkeypatch):
    from app.services import discovery

    fake = FakeRedis()

    async def client():
        return fake

    monkeypatch.setattr(discovery, "get_redis_client", client)
    discovery._l1_project_cache.clear()
    return fake


async def test_a_campaign_is_cleared_under_its_own_key(redis):
    """The key the caller reads is project_context:<campaign id>. Anything else deletes
    nothing at all, and deleting nothing looks exactly like deleting something."""
    from app.services import discovery

    await discovery.invalidate_campaign_context("camp-1")
    assert redis.deleted == ["project_context:camp-1"]


async def test_the_in_process_cache_is_dropped_too(redis):
    """Redis alone is not enough: the L1 dict answers for five minutes without ever asking
    Redis, which is several calls on a busy campaign."""
    from app.services import discovery

    discovery._l1_project_cache["camp-1"] = (0, {"name": "old"})
    await discovery.invalidate_campaign_context("camp-1")
    assert "camp-1" not in discovery._l1_project_cache


async def test_editing_a_project_clears_every_campaign_selling_it(redis):
    """A project is reached through its campaigns, so one edit has to clear all of them."""
    from app.services import discovery

    cleared = await discovery.invalidate_project_everywhere(FakeDb(["camp-1", "camp-2"]), "proj-9")
    assert cleared == 2
    assert redis.deleted == ["project_context:camp-1", "project_context:camp-2"]


async def test_the_project_id_is_never_used_as_a_cache_key(redis):
    """The whole bug in one line: invalidate_project_cache(project_id) matched no key."""
    from app.services import discovery

    await discovery.invalidate_project_everywhere(FakeDb(["camp-1"]), "proj-9")
    assert not any("proj-9" in key for key in redis.deleted)


async def test_a_project_with_no_campaigns_clears_nothing_and_does_not_raise(redis):
    """The delete route reaches this after refusing any project that still has campaigns."""
    from app.services import discovery

    assert await discovery.invalidate_project_everywhere(FakeDb([]), "proj-9") == 0
    assert redis.deleted == []


def test_both_project_routes_clear_the_cache_by_campaign():
    """Structural, because the routes need a database to run. It is the assignment these two
    got wrong for the whole life of the feature."""
    import inspect

    from app.api.routes import dashboard

    for route in (dashboard.update_project, dashboard.delete_project):
        src = inspect.getsource(route)
        assert "invalidate_project_everywhere(db, project_id)" in src, route.__name__
        assert "invalidate_campaign_context(str(project_id))" not in src, route.__name__


def test_the_delete_route_resolves_campaigns_before_removing_the_project():
    """Afterwards there is nothing left to look up, so the order is the whole mechanism."""
    import inspect

    from app.api.routes import dashboard

    src = inspect.getsource(dashboard.delete_project)
    assert src.index("invalidate_project_everywhere") < src.index("delete(Project)")
