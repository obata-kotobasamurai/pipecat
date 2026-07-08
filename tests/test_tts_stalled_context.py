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
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test

_FAKE_AUDIO = b"\x00\x01" * 320
_SAMPLE_RATE = 16000


def _fake_audio_for_seconds(seconds: float) -> bytes:
    sample_count = int(_SAMPLE_RATE * seconds)
    return b"\x00\x01" * sample_count


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


class _TruncatingWSTTSService(TTSService):
    """WebSocket-style TTS that sends all timestamps but short audio."""

    def __init__(
        self,
        *,
        word_times: list[tuple[str, float]],
        audio_seconds: float,
        deliver_done: bool = True,
        **kwargs,
    ):
        super().__init__(
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=False,
            sample_rate=_SAMPLE_RATE,
            stop_frame_timeout_s=0.2,
            **kwargs,
        )
        self._word_times = word_times
        self._audio_seconds = audio_seconds
        self._deliver_done = deliver_done

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        async def _deliver():
            await asyncio.sleep(0.01)
            await self.add_word_timestamps(
                self._word_times,
                context_id=context_id,
                includes_inter_frame_spaces=False,
                pre_merge_tokens=False,
            )
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio=_fake_audio_for_seconds(self._audio_seconds),
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                ),
            )
            if self._deliver_done:
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)

        self.create_task(_deliver(), name=f"truncating_deliver_{context_id}")
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


async def _run_truncation_scenario(
    tts: _TruncatingWSTTSService,
    *,
    sleep: float = 0.4,
) -> tuple:
    return await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame(
                text="hello world tail",
                retry_group_id="retry-utterance-1",
                append_to_context=False,
            ),
            SleepFrame(sleep),
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
async def test_done_with_all_timestamps_and_partial_audio_pushes_truncation_error():
    tts = _TruncatingWSTTSService(
        word_times=[("hello", 0.0), ("world", 0.8), ("tail", 2.0)],
        audio_seconds=0.4,
        error_on_truncated_context=True,
    )

    _, up = await _run_truncation_scenario(tts)

    errors = [f for f in up if isinstance(f, TTSErrorFrame)]
    assert len(errors) == 1
    error = errors[0]
    assert error.text == "world tail"
    assert error.tts_context_id
    assert error.retry_group_id == "retry-utterance-1"
    assert "truncated" in error.error


@pytest.mark.asyncio
async def test_done_with_all_timestamps_and_full_audio_does_not_error():
    tts = _TruncatingWSTTSService(
        word_times=[("hello", 0.0), ("world", 0.8), ("tail", 2.0)],
        audio_seconds=2.4,
        error_on_truncated_context=True,
    )

    _, up = await _run_truncation_scenario(tts)

    assert not [f for f in up if isinstance(f, TTSErrorFrame)]


@pytest.mark.asyncio
async def test_short_utterance_below_truncation_threshold_does_not_error():
    tts = _TruncatingWSTTSService(
        word_times=[("hello", 0.0), ("tail", 0.9)],
        audio_seconds=0.1,
        error_on_truncated_context=True,
    )

    _, up = await _run_truncation_scenario(tts)

    assert not [f for f in up if isinstance(f, TTSErrorFrame)]


@pytest.mark.asyncio
async def test_timeout_with_all_timestamps_and_partial_audio_pushes_truncation_error():
    tts = _TruncatingWSTTSService(
        word_times=[("hello", 0.0), ("world", 0.8), ("tail", 2.0)],
        audio_seconds=0.4,
        deliver_done=False,
        error_on_stalled_context=True,
        error_on_truncated_context=True,
    )

    _, up = await _run_truncation_scenario(tts, sleep=0.8)

    errors = [f for f in up if isinstance(f, TTSErrorFrame)]
    assert len(errors) == 1
    assert errors[0].text == "world tail"
    assert "timeout" in errors[0].error


@pytest.mark.asyncio
async def test_no_stall_error_when_all_words_were_spoken():
    """A timeout with no unspoken text (e.g. lost 'done') must not raise errors."""
    tts = _StallingWSTTSService(deliver_words=3, error_on_stalled_context=True)
    _, up = await _run_stall_scenario(tts)

    assert not [f for f in up if isinstance(f, TTSErrorFrame)]
    assert tts.stalled_context_ids == []
