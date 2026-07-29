#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Per-transcript flush boundaries for the Cartesia websocket service.

Every transcript submitted to one Cartesia context shares that context's
``context_id``, so ``context_id`` alone cannot say *whose* audio has finished
arriving. That matters for any caller that mixes locally-produced audio (TTS
cache hits, recorded digit PCM) with server-streamed audio inside one turn:
server audio arrives asynchronously on the receive loop, so without a
per-transcript completion signal the only safe policy is "once one sentence
goes to the server, stop using local audio for the rest of the turn".

Cartesia's protocol does provide the signal — ``flush``/``flush_done`` with a
1-based ``flush_id`` — the service just never surfaced it. These tests cover
that plumbing:

- a boundary flush keeps the context open (``continue=true``) unlike
  ``flush_audio``, which ends it (``continue=false``)
- ``flush_id`` is allocated per context and increments
- ``flush_done`` releases exactly the waiters it should
- a later ``flush_done`` implies earlier boundaries completed, so a dropped
  message cannot strand a waiter
- context teardown (done / error / interruption) releases waiters instead of
  making them wait out their timeout
- malformed or missing ``flush_id`` is ignored rather than raising on the
  receive loop
"""

import asyncio
import json

import pytest

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.settings import TTSSettings


class _FakeWebsocket:
    """Collects sent frames so tests can assert on the wire format."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, message: str):
        self.sent.append(json.loads(message))


def _service() -> CartesiaTTSService:
    """Build a service without running __init__ (which would open a connection)."""
    service = CartesiaTTSService.__new__(CartesiaTTSService)
    # Logging paths reach for the processor name; __init__ is skipped here.
    service._name = "CartesiaTTSService#test"
    service._settings = TTSSettings(language="ja")
    service._settings.voice = "voice-id"
    service._settings.model = "sonic-3.5"
    service._settings.generation_config = None
    service._settings.pronunciation_dict_id = None
    service._output_container = "raw"
    service._output_encoding = "pcm_s16le"
    service._output_sample_rate = 24000
    service._max_buffer_delay_ms = 0
    service._flush_sent = {}
    service._flush_waiters = {}
    service._websocket = _FakeWebsocket()
    # Contexts the service considers live; wait_for_flush_done checks this.
    service._live_contexts = {"ctx1", "ctx2"}
    service.audio_context_available = lambda cid: cid in service._live_contexts
    return service


# ---------------------------------------------------------------------------
# wire format
# ---------------------------------------------------------------------------


def test_boundary_flush_keeps_context_open():
    """A boundary flush must not end the context.

    ``flush_audio`` sends ``continue=false``, which finalizes the context. A
    boundary flush has to send ``continue=true, flush=true`` so later sentences
    in the same turn still land in the same context.
    """
    service = _service()

    asyncio.run(service.flush_transcript_boundary("ctx1"))

    assert len(service._websocket.sent) == 1
    msg = service._websocket.sent[0]
    assert msg["continue"] is True, "boundary flush must not close the context"
    assert msg["flush"] is True
    assert msg["context_id"] == "ctx1"
    assert msg["transcript"] == ""


def test_normal_generation_message_carries_no_flush_field():
    """Ordinary synthesis must be byte-identical to before this change."""
    service = _service()

    msg = json.loads(service._build_msg(text="こんにちは", context_id="ctx1"))

    assert "flush" not in msg, "flush must be opt-in, never on normal sends"
    assert msg["continue"] is True
    assert msg["transcript"] == "こんにちは"


# ---------------------------------------------------------------------------
# flush_id allocation
# ---------------------------------------------------------------------------


def test_flush_ids_increment_per_context_starting_at_one():
    service = _service()

    first = asyncio.run(service.flush_transcript_boundary("ctx1"))
    second = asyncio.run(service.flush_transcript_boundary("ctx1"))
    other = asyncio.run(service.flush_transcript_boundary("ctx2"))

    assert first == 1, "Cartesia's flush_id is 1-based"
    assert second == 2
    assert other == 1, "counters are per context, not global"


def test_boundary_flush_without_websocket_is_a_noop():
    service = _service()
    service._websocket = None

    assert asyncio.run(service.flush_transcript_boundary("ctx1")) is None
    assert service._flush_sent == {}, "a failed send must not consume an id"


# ---------------------------------------------------------------------------
# flush_done handling
# ---------------------------------------------------------------------------


def test_flush_done_releases_the_matching_waiter():
    service = _service()

    async def scenario():
        flush_id = await service.flush_transcript_boundary("ctx1")
        waiter = asyncio.ensure_future(
            service.wait_for_flush_done("ctx1", flush_id, timeout=5)
        )
        await asyncio.sleep(0)  # let the waiter register
        await service._handle_flush_done("ctx1", flush_id)
        return await waiter

    assert asyncio.run(scenario()) is True


def test_flush_done_does_not_release_a_different_context():
    """Two turns can be in flight; a boundary in one must not free the other."""
    service = _service()

    async def scenario():
        waiter = asyncio.ensure_future(
            service.wait_for_flush_done("ctx1", 1, timeout=0.2)
        )
        await asyncio.sleep(0)
        await service._handle_flush_done("ctx2", 1)
        return await waiter

    assert asyncio.run(scenario()) is False, "waiter woke on the wrong context"


def test_later_flush_done_implies_earlier_boundaries_completed():
    """A dropped flush_done must not strand a waiter for the rest of the turn."""
    service = _service()

    async def scenario():
        first = asyncio.ensure_future(service.wait_for_flush_done("ctx1", 1, timeout=5))
        second = asyncio.ensure_future(service.wait_for_flush_done("ctx1", 2, timeout=5))
        await asyncio.sleep(0)
        # Only the *second* boundary is acknowledged.
        await service._handle_flush_done("ctx1", 2)
        return await first, await second

    assert asyncio.run(scenario()) == (True, True)


def test_flush_done_for_an_earlier_boundary_leaves_later_waiters_blocked():
    service = _service()

    async def scenario():
        pending = asyncio.ensure_future(service.wait_for_flush_done("ctx1", 2, timeout=0.2))
        await asyncio.sleep(0)
        await service._handle_flush_done("ctx1", 1)
        return await pending

    assert asyncio.run(scenario()) is False


@pytest.mark.parametrize("bad_id", [None, "abc", {}, []])
def test_malformed_flush_id_is_ignored(bad_id):
    """The receive loop must never raise on an unexpected payload."""
    service = _service()

    async def scenario():
        await service._handle_flush_done("ctx1", bad_id)

    asyncio.run(scenario())  # must not raise


def test_flush_done_with_string_digits_is_accepted():
    service = _service()

    async def scenario():
        waiter = asyncio.ensure_future(service.wait_for_flush_done("ctx1", 1, timeout=5))
        await asyncio.sleep(0)
        await service._handle_flush_done("ctx1", "1")
        return await waiter

    assert asyncio.run(scenario()) is True


# ---------------------------------------------------------------------------
# teardown: interruption / error / context end
# ---------------------------------------------------------------------------


def test_waiting_on_a_dead_context_returns_immediately():
    """Barge-in tears the context down; a caller must not block on it."""
    service = _service()

    assert asyncio.run(service.wait_for_flush_done("gone", 1, timeout=5)) is False


def test_context_teardown_releases_waiters():
    """On interruption/error the waiter is freed rather than timing out.

    Returning False (ordering no longer guaranteed) is the point — the caller
    falls back instead of hanging for the full timeout mid-call.
    """
    service = _service()

    async def scenario():
        waiter = asyncio.ensure_future(service.wait_for_flush_done("ctx1", 1, timeout=30))
        await asyncio.sleep(0)
        service._discard_flush_state("ctx1")
        # Deliberately short: proves we did not sit out the 30s timeout.
        return await asyncio.wait_for(waiter, timeout=1)

    assert asyncio.run(scenario()) is True


def test_teardown_clears_the_flush_counter():
    """A recycled context id must restart at 1, not continue the old count."""
    service = _service()

    asyncio.run(service.flush_transcript_boundary("ctx1"))
    service._discard_flush_state("ctx1")

    assert asyncio.run(service.flush_transcript_boundary("ctx1")) == 1


def test_teardown_of_one_context_leaves_the_other_intact():
    service = _service()

    asyncio.run(service.flush_transcript_boundary("ctx1"))
    asyncio.run(service.flush_transcript_boundary("ctx2"))
    service._discard_flush_state("ctx1")

    assert "ctx2" in service._flush_sent
    assert asyncio.run(service.flush_transcript_boundary("ctx2")) == 2


def test_timeout_cleans_up_its_waiter():
    """A timed-out wait must not leak state that a later flush_done would touch."""
    service = _service()

    async def scenario():
        result = await service.wait_for_flush_done("ctx1", 1, timeout=0.05)
        return result, dict(service._flush_waiters)

    result, waiters = asyncio.run(scenario())
    assert result is False
    assert waiters == {}, "timed-out waiter was left registered"
