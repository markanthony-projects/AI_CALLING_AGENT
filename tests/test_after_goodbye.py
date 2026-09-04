"""Two things the prospect heard on call 3c43b6bc that they should not have.

Two sign-offs, back to back:

    11:21:45  end_call fires, goodbye queued
    11:21:45  USER  → "No, I don't want to visit the site Thank you, you are
                       repeating question."
    11:21:51  AGENT → "I will send you the brochure and price details on
                       WhatsApp. Thank you for your time, Rahul. Have a great day!"
    11:21:58  AGENT → "No problem at all. I apologize for that. I will send you
                       the brochure, floor plans and price details on WhatsApp..."

And, earlier in the same call, a request the agent had no way to honour:

    USER  → "Sorry I didn't catch that. Can you say it again with"
    USER  → "little bit slow hai?"

It repeated itself twice at exactly the same speed, because the pace was a constant.
"""

import ast
import asyncio
import inspect

import pytest
from pipecat.frames.frames import Frame, InterruptionFrame, LLMContextFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection

from app.utils.closing_gate import ClosingGate
from app.utils.pace import MIN_PACE, PACE_STEP, adjusted_pace, pace_request


# --- nothing gets generated after the goodbye ----------------------------------------------


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _feed(gate, frames):
    import pipecat.processors.frame_processor as fp

    captured = _Captured()
    gate.push_frame = captured.push
    gate.process_frame = gate.__class__.process_frame.__get__(gate)

    async def noop(*a, **kw):
        pass

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        for frame in frames:
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured.frames


def _context_frame():
    from pipecat.processors.aggregators.llm_context import LLMContext

    return LLMContextFrame(context=LLMContext(messages=[{"role": "system", "content": "x"}]))


def test_an_ordinary_turn_passes_straight_through():
    """Every turn of every call goes through this. Holding anything back before end_call
    would be the whole conversation, not the goodbye."""
    gate = ClosingGate("sid")
    passed = asyncio.run(_feed(gate, [_context_frame()]))
    assert len(passed) == 1


def test_a_turn_generated_after_the_goodbye_never_reaches_the_model():
    gate = ClosingGate("sid")
    gate.arm()
    assert asyncio.run(_feed(gate, [_context_frame()])) == []


def test_the_goodbye_itself_still_gets_through():
    """It is queued as a TTSSpeakFrame at the task source and never passes the LLM, but a
    gate that swallowed everything would take the farewell with it — which is the bug this
    exists to prevent, arrived at from the other side."""
    gate = ClosingGate("sid")
    gate.arm()
    passed = asyncio.run(_feed(gate, [TextFrame("Thank you for your time.")]))
    assert len(passed) == 1


def test_there_is_no_way_back_once_it_is_armed():
    """end_call is irreversible and so is this. A reopened gate would let a turn through
    after the carrier leg had already been told to end."""
    gate = ClosingGate("sid")
    gate.arm()
    assert gate.closing is True
    assert not hasattr(gate, "disarm")


def test_the_turns_it_drops_are_counted():
    """Two dropped turns during one goodbye says the prospect kept talking through it, which
    is worth knowing about the goodbye."""
    gate = ClosingGate("sid")
    gate.arm()
    asyncio.run(_feed(gate, [_context_frame(), _context_frame()]))
    assert gate.dropped == 2


def test_it_sits_where_the_frame_it_drops_actually_exists():
    """LLMContextFrame is emitted by the user aggregator and consumed by the LLM. Anywhere
    else in the pipeline the gate would never see one and would do nothing at all."""
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    order = src[src.index("pipeline = Pipeline(["):]
    assert order.index("user_agg,") < order.index("closing_gate,") < order.index("llm,")


@pytest.mark.parametrize("handler", ["end_call_handler", "on_leaked_end_call"])
def test_both_ways_out_of_a_call_arm_it(handler):
    """A model can hang up through the tool or by writing the tool call into its speech.
    Arming only one path leaves the other with the bug."""
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name == handler
    )
    assert "closing_gate.arm()" in ast.unparse(node)


def test_an_interruption_passes_through_before_the_goodbye_is_queued():
    """Every barge-in of every call goes through here, and the one end_call sends itself to
    clear a stale reply goes through here too."""
    gate = ClosingGate("sid")
    gate.arm()
    assert len(asyncio.run(_feed(gate, [InterruptionFrame()]))) == 1


def test_the_prospect_cannot_cut_off_the_goodbye():
    """An interrupted TTSSpeakFrame is never spoken at all. On 4 Sep a caller heard silence
    where the sign-off should have been, then eleven seconds of the farewell wait running out
    against audio that was never coming, and then a dead line."""
    gate = ClosingGate("sid")
    gate.arm()
    gate.protect_goodbye()
    assert asyncio.run(_feed(gate, [InterruptionFrame()])) == []


def test_the_shield_does_not_swallow_the_goodbye_it_exists_to_protect():
    """The closing line is queued at the task source, so it travels the whole pipeline and
    passes through here like everything else. A shield that stopped at "are we protecting?"
    would silence exactly the sentence it was raised for."""
    from pipecat.frames.frames import TTSSpeakFrame

    gate = ClosingGate("sid")
    gate.arm()
    gate.protect_goodbye()
    passed = asyncio.run(_feed(gate, [TTSSpeakFrame("Thank you for your time, Rahul.")]))
    assert len(passed) == 1


def test_talking_over_the_goodbye_is_counted():
    gate = ClosingGate("sid")
    gate.protect_goodbye()
    asyncio.run(_feed(gate, [InterruptionFrame(), InterruptionFrame()]))
    assert gate.shielded == 2


def test_the_shield_is_not_raised_by_arming_alone():
    """Two separate moments. end_call opens by interrupting a stale reply from a split turn,
    and a shield raised at arm() would swallow the pipeline's own interruption — sent by us,
    three lines before the goodbye."""
    gate = ClosingGate("sid")
    gate.arm()
    assert gate.protecting is False


def test_shielding_does_not_also_start_letting_new_turns_through():
    """The two guards are independent, and the wrong one lifting would put a fresh reply
    behind the goodbye — which is the bug the gate was written for."""
    gate = ClosingGate("sid")
    gate.arm()
    gate.protect_goodbye()
    assert asyncio.run(_feed(gate, [_context_frame()])) == []


# --- speaking slower when asked -------------------------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "little bit slow hai?",
        "Can you speak slowly",
        "please speak a bit slower",
        "thoda dheere boliye",
        "dhire bolo na",
        "aaram se boliye",
        "you are speaking too fast",
        "धीरे बोलिए",
    ],
)
def test_a_request_to_slow_down_is_recognised(said):
    assert pace_request(said) == "slower"


@pytest.mark.parametrize("said", ["you are too slow", "speed it up", "jaldi boliye", "can you go faster"])
def test_a_request_to_speed_up_is_recognised(said):
    """"Too slow" contains the word "slow" and would otherwise be read as a request to slow
    down further, which is the opposite of what was said."""
    assert pace_request(said) == "faster"


@pytest.mark.parametrize(
    "said",
    [
        "Yeah, I was looking for property.",
        "The 3.5 BHK is what in my budget?",
        "Maybe around in next 6 months.",
        "",
        None,
    ],
)
def test_an_ordinary_turn_changes_nothing(said):
    assert pace_request(said) is None


def test_one_step_per_request():
    """A prospect who is still not comfortable asks again. Two steps from one sentence would
    leave the agent crawling."""
    assert adjusted_pace(1.0, "slower", 1.0) == pytest.approx(1.0 - PACE_STEP)


def test_asking_twice_slows_it_twice():
    once = adjusted_pace(1.0, "slower", 1.0)
    assert adjusted_pace(once, "slower", 1.0) < once


def test_it_stops_at_the_floor_however_many_times_they_ask():
    pace = 1.0
    for _ in range(20):
        pace = adjusted_pace(pace, "slower", 1.0)
    assert pace == MIN_PACE


def test_the_floor_is_a_speed_sarvam_will_accept():
    """Pipecat forwards this into the config with no range check, and an out-of-range value
    has already killed TTS for a whole call once. Read from the model's own table rather
    than restated here, so a model swap cannot leave the floor below what it takes."""
    from pipecat.services.sarvam.tts import TTS_MODEL_CONFIGS

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    model = next(m for m in TTS_MODEL_CONFIGS if f'model="{m}"' in src)
    low, high = TTS_MODEL_CONFIGS[model].pace_range
    assert low <= MIN_PACE <= high, f"{model} accepts {low} to {high}"


def test_the_floor_is_still_a_person_speaking_carefully():
    """Sarvam accepts far slower than this. A judgement rather than a limit: somewhere below
    about 0.7 the voice stops sounding like someone taking their time and starts sounding
    broken, which is not what "please speak slowly" asked for. Pinned so that moving it is a
    decision somebody made rather than a number that drifted."""
    assert MIN_PACE >= 0.7


def test_faster_only_walks_back_a_slow_down():
    """Nobody on a cold sales call wants the pitch delivered faster than it was written. A
    request to speed up returns towards where the call started and stops there."""
    assert adjusted_pace(1.0, "faster", 1.0) == 1.0
    slowed = adjusted_pace(1.0, "slower", 1.0)
    assert adjusted_pace(slowed, "faster", 1.0) == 1.0


def test_saying_nothing_about_it_leaves_the_pace_alone():
    assert adjusted_pace(0.85, None, 1.0) == 0.85


# --- and how the call uses it ----------------------------------------------------------------


def _agent_source():
    from app.services import agent

    return inspect.getsource(agent.run_voice_agent)


def test_the_change_reaches_the_voice_engine_without_a_reconnect():
    """TTSUpdateSettingsFrame reaches Sarvam's _update_settings, which resends the config on
    the connection already open. Rebuilding the service instead would drop the websocket —
    and a reconnect mid-call is exactly what left the agent mute on 4 Sep."""
    src = _agent_source()
    assert "TTSUpdateSettingsFrame(delta=SarvamTTSService.Settings(pace=_pace))" in src


def test_it_is_read_off_what_the_prospect_actually_said():
    src = _agent_source()
    assert "pace_request(transcript)" in src


def test_nothing_is_sent_when_nothing_changed():
    """This fires on every turn of every call. Pushing a settings frame each time would put
    a config round trip on the turn path for no reason."""
    src = _agent_source()
    assert "if wanted != _pace:" in src


def test_the_ceiling_is_the_pace_the_call_opened_with():
    from app.services import agent

    assert "adjusted_pace(_pace, pace_request(transcript), SPEAKING_PACE)" in _agent_source()
    assert agent.SPEAKING_PACE == 1.0
