# Decisions

What was tried, what was kept, and why. Newest first within each area.

Each entry is one decision. If it was driven by a live call, the call id is there — the full
story lives in the code comment and the test, this is the index.

**The rule this file exists to record:** when the prompt tells the model to do something and
it does not, twice, the third fix belongs in code. Guards beat prose. Every entry marked
CODE below started as a prompt rule that was ignored on a live call.

---

## Speech-to-text

**Deepgram Flux, not nova-2.** (`c4582dc`, 11 Sep)
Flux decides end-of-turn on Deepgram's side instead of us guessing with a stopwatch. Chosen
by a measured bake-off across three calls, not by preference. Its Hindi is also clearly
better: nova-2 heard "1.4" as "4 5 CR" and the agent hung up on the correction.

**`STT_EOT_THRESHOLD=0.5`, not the 0.7 default.** (12 Sep, env)
At 0.7 `turn_decision` was 616ms — no better than the stopwatch it replaced. At 0.5 it is
~440-620ms, and a deliberately paused sentence ("My budget is… one point four five") still
came through whole.

**The first STT connect is started, not awaited.** (`487a4c9`, 12 Sep) — **CODE**
StartFrame walks the pipeline one processor at a time, and `DeepgramFluxSTTBase.start` awaits
its own websocket handshake. STT sits before TTS, so the voice engine could not begin its own
172ms connect until STT had finished its ~800ms one — and the greeting waits for all of them.
Backgrounding the first connect took `FIRST WORD` from 2324ms to 1446ms. Reconnects are still
awaited: by then the caller is handling a live failure and "connected" must mean connected.

**A turn the service never ends is ended for it.** (`781077a`, 12 Sep) — **CODE**
Call 7b00a8af: Flux held one turn open for 43 seconds while the prospect said "Hello?" into
silence, then hung up. Pipecat's own backstop skips `if self._user_speaking`, which with Flux
is true the entire time a turn is stuck. Ceiling is 10s — too long for a real sentence, far
too short for that.

**Sarvam STT is not a candidate.** (10 Sep)
It never sets `finalized=True` in pipecat 1.5, so every turn pays ~0.97s no setting can
remove. Do not re-run that comparison without a pipecat upgrade.

**Smart Turn was never possible.** (11 Sep)
`turn_analyzer` is not a field on `TransportParams` in pipecat 1.5. `SMART_TURN_ENABLED`
wires an object to nothing. Should be made real or deleted.

---

## Turn-taking and barge-in

**Two gates, chosen by what the service can tell us.** (`b7288d6`, 11 Sep) — **CODE**
A word gate needs interim transcripts. Flux and Sarvam emit none, so with them the agent
talked over the caller's whole utterance. `SustainedSpeechBargeIn` counts seconds instead
(0.8s). Which gate a call gets is decided in `stt_provider` next to the service, because it
follows from that choice.

0.8s is arithmetic, not taste: VAD reports speech stopped 0.2s late, so a half-second
"hello?" cancels the clock at ~0.7s and must not have fired first.

**Agreeing along does not interrupt.** (`b7288d6`, 11 Sep) — **CODE**
"haan haan", "achha", "theek hai", "हाँ" are how a listener shows they are there. They are
held, not discarded — under `min_words` the base class was throwing them away entirely.
"nahi" and "no" are deliberately absent: a refusal is never a backchannel.

**"Ruko" means stop.** (`fb71410`, `2a89228`, 11 Sep) — **CODE**
Call 8d86156e: the prospect asked four times, in two languages, and the agent said "Sure,
I'll wait" and then repeated its pitch. The words were right and the behaviour was wrong —
a reply is the only thing a model can emit, so silence has to be the code's decision.

Two lessons in the pattern itself: "एक minute" mixes scripts inside one phrase, and "One one
minute" stutters. Both missed the first draft. **People repeat and stutter when they want you
to stop; the repetition is the request.**

**"Hello?" is not permission to pitch.** (`78f9e1e`, `65ea6fd`, 11-12 Sep) — **CODE**
Three calls answered a one-word line check with thirty words of project. The re-introduction
drops the time of day ("Good afternoon" twice in ten seconds is the most automated thing a
caller can hear), and does not fire while the greeting is still playing — they are hearing it.

---

## The brain

**gemma-4-31b is gone.** (10 Sep)
Cerebras withdrew it on 3 Sep; it 404ed on 10 Sep and every call failed. A fallback model
must always be configured.

**`LLM_REASONING_EFFORT=medium`, not low.** (12 Sep, env) — **the fix for "the agent went dumb"**
At `low` (12-106 reasoning tokens) gpt-oss-120b produced degenerate output on live calls:
four turns of conversation in one reply including the prospect's invented answer, and the
same question repeated four times with no separators. At `medium` (94-866 tokens) both
stopped. Costs ~257ms p50 and is worth it — a fast agent that talks to itself is worthless.

**What `unattributed` in the latency line actually is.** (12 Sep)
`unattributed ≈ 490ms + 0.67ms per reasoning token`, fitted across both effort settings. It
is the model finishing its first sentence, plus a floor of TTS audio start. Shortening the
first sentence is therefore a latency fix, not only a style one.

---

## Voice (TTS)

**A failed reconnect no longer mutes the call.** (`4e3352e`, 12 Sep) — **CODE**
Root cause of call 5023ff25, read out of pipecat rather than guessed:
`InterruptibleTTSService` disconnects and reconnects on every barge-in; Sarvam's connect sets
`_websocket = None` on failure; `_get_websocket` then raises forever. One failed reconnect and
the agent is mute for the rest of the call. The socket is now checked before the next thing
the agent says — not on the error event, because that arrives mid-barge-in, which is the race
that lost it. Bounded at 3 per call.

**Three reconnects in one call is a WARNING.** (`4e3352e`, 12 Sep)
Four preceded the mute, and every one was logged at INFO among a thousand INFO lines.

**`SARVAM_TEMPERATURE=0.4`.** (12 Sep, env) — judged by ear, not measured.

**Speaking pace stays at default.** (10 Sep)
1.1 and 1.05 were both tried on live calls and sounded too fast.

---

## The prompt

**A prompt is instructions, not a design document.** (`01d46a6`, 12 Sep)
The prompt had been written like code comments — every rule carried with the live call that
caused it and the dialogue quoted as evidence. It reached 29,376 chars and 55 quoted
conversational fragments. The evidence belongs in the tests, which is where it now lives;
`tests/test_call_script.py` carries the calls and asserts on the imperative.

Measured afterwards: prompt size costs ~18ms of TTFB per 1000 tokens. **Size was never the
problem. Density on one topic was.**

**Seven rules on one sentence made it worse, not better.** (`28f7022`, 12 Sep) — **CODE**
The step-2 headline took three prompt attempts — "must have a verb", then "two sentences must
not start the same way", then "vary by rewriting, never by deleting the verb" — and each
attempt added a constraint the model had to satisfy simultaneously. It dropped whichever it
dropped. The sentence is built in `app/utils/headline.py` now and arrives finished. The prompt
lost 1796 characters and gained 726.

**Per-call lines go at the bottom.** (`f0eb13c`, 10 Sep)
Static rulebook first, `name_line` and campaign context last. Prompt cache went 5% → 98%.
Never put anything per-call above the static block.

**One question per turn, and react before the next one.** (`65ea6fd`, `44f3951`, 11-12 Sep)
A list of five example questions under "one question per turn" was read out as a list — all
five in one breath, and the prospect said so. Fixing that produced four bare questions in a
row, which is an interview. Both halves are needed: one question, and three words about what
they just said.

**A decision is not a guess.** (`44f3951`, 12 Sep)
"Not this area" from someone never told where the project is, is a guess worth correcting —
that rule saved a 1.5 Cr lead. "Not in this project" from someone who has heard it all is an
answer. Stop pitching; raise it once more only if their stated requirement genuinely fits,
in one sentence, with their own words in it.

**Read back before hanging up.** (`44f3951`, 12 Sep)
Every close, not only the rejection path. The read-back is the only proof the prospect has
that any of it was heard.

---

## Measurement

**Time the turn from when their voice stopped.** (`67789f9`, 10 Sep)
The metric was called voice-to-voice and started 600ms late. This is the change that made
every later finding visible — the 600ms turn window, the 800ms STT handshake, the reasoning
correlation. **Measure honestly before optimising anything.**

**`STARTUP` splits the silence before the greeting.** (`0e9d32f`, `daa9185`, 12 Sep)
Four marks off the stream-open clock: services built, stt, tts, pipeline, greeting queued.
Reported as gaps, because the question is never "when did TTS connect" — it is what the
greeting was waiting for.

**A log line nobody can see is not instrumentation.** (`daa9185`, 12 Sep)
`app/main.py` filters INFO to an allow-list. Three separate instruments shipped invisible
because their module was not in it. The rule is derived now, not listed:
`tests/test_log_noise.py` walks `app/` for any `logger.info` carrying a call_sid and asserts
its module is in the set.

---

## Testing

**Mutation testing, every change.** (throughout)
Write the change, write the tests, then break the change on purpose and check the tests
notice. It has caught a weak test on almost every commit in this file — assertions on
docstrings instead of behaviour, ceilings tested with a wait shorter than the ceiling, tests
pinning a prompt's narrative rather than its rule.

**Never stub the contract you are testing.** (`4e3352e` aftermath, 12 Sep)
`KeepsItsVoice.run_tts` declared `(self, text)` while pipecat's is `(self, text, context_id)`.
Every call raised TypeError and production ended a call 1.2s in with "tts unavailable". The
suite was green because its fakes had copied the wrong signature — it was checking the stub
against itself.

`tests/test_override_signatures.py` now finds every class under `app/` that subclasses
pipecat and binds the base's parameters to ours. A pipecat version bump cannot silently break
an override any more.

**One change per deploy is the wrong rule.** (12 Sep)
The right one: two changes may ship together when they cannot produce the same symptom. A
guard in the opening two seconds and a prompt rule that only runs after a rejection are
separable by any live call. An LLM change and a turn-threshold change are not — both move
latency and the logs cannot tell them apart.
