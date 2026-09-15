"""The sample rate the voice engine is asked for, in one place.

Pipecat's SarvamTTSService asks Sarvam for 24kHz audio — its constructor default, kept
whatever the transport plays at — and labels the frames it pushes with that same rate, so
the output transport resamples them to the 16kHz the carrier takes. Every live line on
every call has travelled that way.

Call 56398497, 15 Sep 2026, is why this is written down. A socket warmed before the media
stream was configured for 16kHz "to match the transport"; the service then started, took
its rate from the constructor as it always does, and labelled the 16kHz audio as 24kHz.
The transport resampled accordingly and the prospect heard every line after the greeting
1.5 times too fast and seven semitones too high, until a barge-in opened a fresh socket at
the right rate. The greeting, synthesised at 16kHz and labelled 16kHz, was the only line
that sounded right — which is what pointed at the cause.

So: one number, imported by everything that opens a Sarvam socket or plays cached audio
into the pipeline, and a test that holds it equal to what the service will actually use.
"""

SAMPLE_RATE = 24000
