"""Every method this repository overrides must accept what its base is called with.

Production broke on 12 Sep 2026 because KeepsItsVoice.run_tts declared (self, text) while
SarvamTTSService.run_tts is (self, text, context_id). Every call raised TypeError, three of
them tripped MAX_TTS_FAILURES, and the call ended 1.2 seconds in with "tts unavailable".

The suite was green. Its fakes had copied the wrong signature, so it was checking the stub
against itself and agreeing with the bug.

The class of error is bigger than that one method. This repository subclasses a lot of
pipecat — speech services, turn strategies, frame processors — to change one small thing
each time, and every one of those overrides is a signature that pipecat owns and can move.
A version bump is enough to break any of them, silently, until a live call.

So the check is derived rather than listed: find every class under app/ that subclasses
something from pipecat, and for each method it overrides, bind the BASE's parameters to
OURS. Anything the base can be called with, ours has to take.
"""

import importlib
import inspect
from pathlib import Path

import pytest


def _app_modules():
    """Every module under app/, walked from the filesystem.

    pkgutil.walk_packages was tried first and returned two modules — it does not recurse
    here — which would have made this whole file a test that checks nothing and says so
    only in test_there_is_something_to_check.
    """
    for path in sorted(Path("app").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        yield ".".join(parts)


def _our_pipecat_subclasses():
    """Every class under app/ that inherits from pipecat, with the base it overrides."""
    found = []
    for name in _app_modules():
        try:
            module = importlib.import_module(name)
        except Exception:
            continue  # a module that needs a live service is not what this is checking
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not obj.__module__.startswith("app."):
                continue
            bases = [b for b in obj.__mro__[1:] if b.__module__.startswith("pipecat")]
            if bases:
                found.append((obj, bases))
    # de-duplicate: a class re-exported by two modules is one class
    seen, unique = set(), []
    for cls, bases in found:
        key = (cls.__module__, cls.__qualname__)
        if key not in seen:
            seen.add(key)
            unique.append((cls, bases))
    return unique


def _overrides(cls, bases):
    """(name, ours, theirs) for each method this class redefines from a pipecat base."""
    out = []
    for name, ours in vars(cls).items():
        if name.startswith("__") or not callable(ours):
            continue
        for base in bases:
            theirs = getattr(base, name, None)
            if theirs is None or not callable(theirs):
                continue
            if getattr(theirs, "__module__", "").startswith("pipecat"):
                out.append((name, ours, theirs))
            break
    return out


def test_there_is_something_to_check():
    """Guards the discovery: a walk that silently found nothing would pass forever."""
    classes = _our_pipecat_subclasses()
    names = {c.__name__ for c, _ in classes}
    assert "KeepsItsVoice" in names, names
    assert len(classes) >= 4, names


@pytest.mark.parametrize(
    "cls,bases", _our_pipecat_subclasses(), ids=lambda v: getattr(v, "__name__", "")
)
def test_every_override_accepts_what_its_base_is_called_with(cls, bases):
    for name, ours, theirs in _overrides(cls, bases):
        base_sig = inspect.signature(theirs)
        our_sig = inspect.signature(ours)
        positional = [
            p.name
            for p in base_sig.parameters.values()
            if p.name != "self"
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and p.default is inspect.Parameter.empty
        ]
        try:
            our_sig.bind(object(), *positional)
        except TypeError as exc:
            pytest.fail(
                f"{cls.__module__}.{cls.__name__}.{name}{our_sig} cannot be called the way "
                f"{theirs.__qualname__}{base_sig} is: {exc}"
            )


@pytest.mark.parametrize(
    "cls,bases", _our_pipecat_subclasses(), ids=lambda v: getattr(v, "__name__", "")
)
def test_a_generator_override_is_still_a_generator(cls, bases):
    """The other half of the same mistake, and just as quiet: an override that returns a
    coroutine where the base yields is a method nothing can iterate."""
    for name, ours, theirs in _overrides(cls, bases):
        if inspect.isasyncgenfunction(theirs):
            assert inspect.isasyncgenfunction(ours), (
                f"{cls.__name__}.{name} must be an async generator; "
                f"{theirs.__qualname__} is one"
            )


# --- and the name the metric is filed under ---------------------------------------------


def test_a_subclass_does_not_rename_the_latency_metric():
    """app/utils/latency.py derives its label from the instance name, so wrapping a vendor
    service renames its number. The first call on KeepsItsVoice logged

        keepsitsvoice=201ms  keepsitsvoice_audio=348ms(silence=147ms)

    where every earlier call in this repository logged sarvam=. Nothing failed; the history
    simply stopped matching, which is worse than failing."""
    from app.utils.latency import _short

    # The most derived vendor base only. Several classes in a service's MRO end in
    # "TTSService" — InterruptibleTTSService, WebsocketTTSService — and none of those is the
    # vendor whose latency the number belongs to.
    wrapped = []
    for cls, bases in _our_pipecat_subclasses():
        vendor = next(
            (b for b in bases if b.__name__.endswith(("TTSService", "STTService", "LLMService"))),
            None,
        )
        if vendor is not None:
            wrapped.append((cls, vendor))
    assert wrapped, "no vendor services are wrapped; this test would prove nothing"
    for cls, vendor in wrapped:
        src = inspect.getsource(cls.__init__)
        assert 'setdefault("name"' in src, (
            f"{cls.__name__} does not name itself, so its latency line will read "
            f"'{_short(cls.__name__)}=' instead of '{_short(vendor.__name__)}='"
        )
        assert vendor.__name__ in src, (
            f"{cls.__name__} names itself something other than {vendor.__name__}"
        )
