#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TTS retry-group propagation across synthesis attempts."""

import unittest
from typing import AsyncGenerator

from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    Frame,
    TTSErrorFrame,
    TTSSpeakFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test


class MockErrorTTSService(TTSService):
    """A TTS service that always fails via the base ErrorFrame -> TTSErrorFrame path."""

    def __init__(self, **kwargs):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=False,
            sample_rate=16000,
            **kwargs,
        )

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield ErrorFrame(error=f"synthetic failure for {text}")


class TestTTSRetryGroup(unittest.IsolatedAsyncioTestCase):
    async def test_tts_error_preserves_explicit_retry_group_id(self):
        service = MockErrorTTSService(name="tts_retry_group_test")

        down_frames, up_frames = await run_test(
            service,
            frames_to_send=[TTSSpeakFrame(text="hello", retry_group_id="rg-1")],
            expected_down_frames=None,
            expected_up_frames=[TTSErrorFrame],
        )

        self.assertTrue(any(isinstance(frame, AggregatedTextFrame) for frame in down_frames))
        error_frame = up_frames[0]
        self.assertEqual(error_frame.retry_group_id, "rg-1")
        self.assertIsNotNone(error_frame.tts_context_id)
        self.assertEqual(service.get_audio_contexts(), [])
        self.assertEqual(service._retry_group_by_context, {})

    async def test_tts_error_defaults_retry_group_id_to_context_id(self):
        service = MockErrorTTSService(name="tts_retry_group_default")

        _, up_frames = await run_test(
            service,
            frames_to_send=[TTSSpeakFrame(text="hello")],
            expected_down_frames=None,
            expected_up_frames=[TTSErrorFrame],
        )

        error_frame = up_frames[0]
        self.assertEqual(error_frame.retry_group_id, error_frame.tts_context_id)
        self.assertIsNotNone(error_frame.tts_context_id)
        self.assertEqual(service._retry_group_by_context, {})

    async def test_multiple_attempts_can_share_retry_group_with_distinct_contexts(self):
        service = MockErrorTTSService(name="tts_retry_group_reuse")

        _, up_frames = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="hello", retry_group_id="stable-group"),
                SleepFrame(sleep=0.05),
                TTSSpeakFrame(text="hello", retry_group_id="stable-group"),
            ],
            expected_down_frames=None,
            expected_up_frames=[TTSErrorFrame, TTSErrorFrame],
        )

        first_error, second_error = up_frames
        self.assertEqual(first_error.retry_group_id, "stable-group")
        self.assertEqual(second_error.retry_group_id, "stable-group")
        self.assertNotEqual(first_error.tts_context_id, second_error.tts_context_id)
        self.assertEqual(service._retry_group_by_context, {})


if __name__ == "__main__":
    unittest.main()
