#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Lazy STT proxy that defers service creation until the first audio frame arrives."""

from collections.abc import Callable
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.stt_service import STTService


class LazySTTProxy(FrameProcessor):
    """A proxy that lazily initializes an STT service on the first audio frame.

    Useful as a backup in a ServiceSwitcher with FailoverStrategy: the backup
    STT service is not created (and does not open a WebSocket connection) until
    it actually receives audio after a failover switch.

    Example::

        backup_stt_proxy = LazySTTProxy(factory=lambda: GoogleSTTService(...))

        switcher = ServiceSwitcher(
            services=[primary_stt, backup_stt_proxy],
            strategy_type=ServiceSwitcherStrategyFailover,
        )
    """

    def __init__(self, factory: Callable[[], STTService], **kwargs):
        """Initialize the lazy STT proxy.

        Args:
            factory: A callable that creates and returns an STTService instance.
                Called at most once, when the first audio frame arrives.
            **kwargs: Additional arguments passed to FrameProcessor.
        """
        super().__init__(**kwargs)
        self._factory = factory
        self._stt: STTService | None = None
        self._start_frame: StartFrame | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame, lazily initializing the STT service on first audio.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._start_frame = frame
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            if self._stt:
                await self._stt.process_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)
            return

        if isinstance(frame, AudioRawFrame) and self._stt is None:
            await self._initialize_stt()

        if self._stt is not None:
            await self._stt.process_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up the internal STT service if it was created."""
        await super().cleanup()
        if self._stt:
            await self._stt.cleanup()

    async def _initialize_stt(self):
        """Create the STT service and initialize it with the saved StartFrame."""
        logger.info(f"{self} lazily initializing STT service")
        self._stt = self._factory()
        # Set up the internal STT with the same pipeline infrastructure
        setup = FrameProcessorSetup(
            clock=self._clock,
            task_manager=self._task_manager,
            observer=self._observer,
        )
        await self._stt.setup(setup)
        # Wire the internal STT into the pipeline
        self._stt._next = self._next
        self._stt._prev = self._prev
        # Initialize with the saved StartFrame
        if self._start_frame:
            await self._stt.process_frame(self._start_frame, FrameDirection.DOWNSTREAM)
