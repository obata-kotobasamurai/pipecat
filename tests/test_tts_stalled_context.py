#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for mid-stream TTS stall detection (``error_on_stalled_context``).

When a playing audio context stops receiving audio while word-timestamp slots
still hold unspoken text (e.g. the provider websocket died without a close
frame), ``_handle_audio_context`` hits its idle timeout. With
``error_on_stalled_context=True`` the service must:

- invoke the ``on_audio_context_stalled()`` hook (provider connection reset), then
- push a ``TTSErrorFrame`` upstream carrying the remaining unspoken text so
  retry/failover logic (e.g. a ServiceSwitcher strategy) can re-speak it.

With the flag off (default), or when all registered words were spoken before the
timeout (a normal lost-"done" case), the timeout must stay silent as before.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSErrorFrame,
    TTSSpeakFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


class _StallingWSTTSService(TTSService):
    """WebSocket-style word-timestamp TTS that stalls mid-utterance.

    Delivers word timestamps for only the first ``deliver_words`` words plus one
    audio frame, then goes silent: no further audio, no TTSStoppedFrame, no
    remove_audio_context — exactly what a provider websocket dying without a
    close frame looks like to ``_handle_audio_context``.
    """

    def __init__(self, deliver_words: int = 1, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=0.2,
            **kwargs,
        )
        self._deliver_words = deliver_words
        self.stalled_context_ids: list[str] = []

    def can_generate_metrics(self) -> bool:
        return False

    async def on_audio_context_stalled(self, context_id: str):
        self.stalled_context_ids.append(context_id)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver():
            await asyncio.sleep(0.01)
            word_times = [(w, i * 0.1) for i, w in enumerate(text.split())]
            await self.add_word_timestamps(
                word_times[: self._deliver_words],
                context_id=context_id,
                includes_inter_frame_spaces=False,
                pre_merge_tokens=False,
            )
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_FAKE_AUDIO,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            # Stall: never complete the context.

        self.create_task(_deliver(), name=f"stall_deliver_{context_id}")
        if False:
            yield


async def _run_stall_scenario(tts: _StallingWSTTSService) -> tuple:
    # SleepFrame keeps the EndFrame away long enough for the 0.2s
    # audio-context idle timeout to fire and be processed.
    return await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(text="hello world tail", append_to_context=False),
            SleepFrame(0.8),
        ],
    )


@pytest.mark.asyncio
async def test_stalled_context_pushes_tts_error_with_remaining_text():
    tts = _StallingWSTTSService(deliver_words=1, error_on_stalled_context=True)
    _, up = await _run_stall_scenario(tts)

    errors = [f for f in up if isinstance(f, TTSErrorFrame)]
    assert len(errors) == 1, f"Expected exactly one TTSErrorFrame, got {len(errors)}"
    error = errors[0]
    # "hello" was spoken; the unspoken tail must be carried for retry.
    assert "world" in error.text and "tail" in error.text, (
        f"Remaining text must contain the unspoken words, got {error.text!r}"
    )
    assert "hello" not in error.text, (
        f"Already-spoken words must not be retried, got {error.text!r}"
    )
    assert error.tts_context_id, "Stall error must carry the context id"
    # The provider hook must have fired for the same context.
    assert tts.stalled_context_ids == [error.tts_context_id]


@pytest.mark.asyncio
async def test_stall_detection_disabled_by_default():
    tts = _StallingWSTTSService(deliver_words=1)
    _, up = await _run_stall_scenario(tts)

    assert not [f for f in up if isinstance(f, TTSErrorFrame)]
    assert tts.stalled_context_ids == []


@pytest.mark.asyncio
async def test_no_stall_error_when_all_words_were_spoken():
    """A timeout with no unspoken text (e.g. lost 'done') must not raise errors."""
    tts = _StallingWSTTSService(deliver_words=3, error_on_stalled_context=True)
    _, up = await _run_stall_scenario(tts)

    assert not [f for f in up if isinstance(f, TTSErrorFrame)]
    assert tts.stalled_context_ids == []
