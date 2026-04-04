#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio text-to-speech service implementation.

This module provides integration with Fish Audio's real-time TTS WebSocket API
for streaming text-to-speech synthesis with customizable voice parameters.
"""

import asyncio
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSentenceBoundaryFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import ormsgpack
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Fish Audio, you need to `pip install pipecat-ai[fish]`.")
    raise Exception(f"Missing module: {e}")

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "pcm", "wav"]


@dataclass
class FishAudioTTSSettings(TTSSettings):
    """Settings for FishAudioTTSService.

    Parameters:
        latency: Latency mode ("normal" or "balanced"). Defaults to "balanced".
        normalize: Whether to normalize audio output. Defaults to True.
        temperature: Controls randomness in speech generation (0.0-1.0).
        top_p: Controls diversity via nucleus sampling (0.0-1.0).
        prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
        prosody_volume: Volume adjustment in dB (-20 to 20). Defaults to 0.
    """

    latency: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    normalize: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_volume: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> Self:
        """Construct settings from a plain dict, destructuring legacy nested ``prosody``."""
        flat = dict(settings)
        nested = flat.pop("prosody", None)
        if isinstance(nested, dict):
            flat.setdefault("prosody_speed", nested.get("speed"))
            flat.setdefault("prosody_volume", nested.get("volume"))
        return super().from_mapping(flat)


class FishAudioTTSService(InterruptibleTTSService):
    """Fish Audio text-to-speech service with WebSocket streaming.

    Provides real-time text-to-speech synthesis using Fish Audio's WebSocket API.
    Supports various audio formats, customizable prosody controls, and streaming
    audio generation with interruption handling.
    """

    Settings = FishAudioTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Fish Audio TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=FishAudioTTSService.Settings(...)`` instead.

        Parameters:
            language: Language for synthesis. Defaults to English.
            latency: Latency mode ("normal" or "balanced"). Defaults to "normal".
            normalize: Whether to normalize audio output. Defaults to True.
            prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
            prosody_volume: Volume adjustment in dB. Defaults to 0.
        """

        language: Language | None = Language.EN
        latency: str | None = "normal"  # "normal" or "balanced"
        normalize: bool | None = True
        prosody_speed: float | None = 1.0  # Speech speed (0.5-2.0)
        prosody_volume: int | None = 0  # Volume adjustment in dB

    def __init__(
        self,
        *,
        api_key: str,
        reference_id: str | None = None,  # This is the voice ID
        model_id: str | None = None,
        output_format: FishAudioOutputFormat = "pcm",
        sample_rate: int | None = None,
        inter_utterance_silence_s: float = 0.0,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Fish Audio TTS service.

        Args:
            api_key: Fish Audio API key for authentication.
            reference_id: Reference ID of the voice model to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(voice=...)`` instead.

            model_id: Specify which Fish Audio TTS model to use (e.g. "s1").

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(model=...)`` instead.

            output_format: Audio output format. Defaults to "pcm".
            sample_rate: Audio sample rate. If None, uses default.
            inter_utterance_silence_s: Seconds of silence to insert between
                sentences within a single turn. Fish Audio returns one audio
                chunk per flush, so gaps between chunks indicate sentence
                boundaries. Defaults to 0 (no silence).
            params: Additional input parameters for voice customization.

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent service.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="s2-pro",
            voice=None,
            language=None,
            latency="balanced",
            normalize=True,
            temperature=None,
            top_p=None,
            prosody_speed=1.0,
            prosody_volume=0,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if reference_id is not None:
            self._warn_init_param_moved_to_settings("reference_id", "voice")
            default_settings.voice = reference_id
        if model_id is not None:
            self._warn_init_param_moved_to_settings("model_id", "model")
            default_settings.model = model_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.latency is not None:
                    default_settings.latency = params.latency
                if params.normalize is not None:
                    default_settings.normalize = params.normalize
                if params.prosody_speed is not None:
                    default_settings.prosody_speed = params.prosody_speed
                if params.prosody_volume is not None:
                    default_settings.prosody_volume = params.prosody_volume

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = "wss://api.fish.audio/v1/tts/live"
        self._websocket = None
        self._receive_task = None
        self._reconnect_task = None
        self._inter_utterance_silence_s = inter_utterance_silence_s
        logger.debug(f"{self}: inter_utterance_silence_s={self._inter_utterance_silence_s}")

        # Internal retry: on Fish synthesis errors, reconnect WS and resend
        # pending texts without closing the audio context, preserving
        # serialization queue ordering.
        self._pending_texts: dict[str, list[str]] = {}
        self._retry_counts: dict[str, int] = {}
        self._max_internal_retries: int = 3
        self._reconnect_event = asyncio.Event()
        self._reconnect_event.set()

        # Init-only audio format config (not runtime-updatable).
        self._fish_sample_rate = 0  # Set in start()
        self._output_format = output_format

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Fish Audio service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if needed.

        Any change to voice or model triggers a WebSocket reconnect.

        Args:
            delta: A :class:`TTSSettings` (or ``FishAudioTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def start(self, frame: StartFrame):
        """Start the Fish Audio TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._fish_sample_rate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Fish Audio TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Fish Audio TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._reconnect_task:
            await self.cancel_task(self._reconnect_task)
            self._reconnect_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Fish Audio")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            model = assert_given(self._settings.model)
            if model is not None:
                headers["model"] = model
            websocket = await websocket_connect(self._base_url, additional_headers=headers)
            self._websocket = websocket

            # Send initial start message with ormsgpack
            request_settings = {
                "sample_rate": self._fish_sample_rate,
                "latency": self._settings.latency,
                "format": self._output_format,
                "normalize": self._settings.normalize,
                "prosody": {
                    "speed": self._settings.prosody_speed,
                    "volume": self._settings.prosody_volume,
                },
                "reference_id": self._settings.voice,
            }
            if self._settings.temperature is not None:
                request_settings["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                request_settings["top_p"] = self._settings.top_p
            start_message = {"event": "start", "request": {"text": "", **request_settings}}
            await websocket.send(ormsgpack.packb(start_message))
            logger.debug("Sent start message to Fish Audio")

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Fish Audio")
                # Send stop event with ormsgpack
                stop_message = {"event": "stop"}
                await self._websocket.send(ormsgpack.packb(stop_message))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def flush_audio(self, context_id: str | None = None):
        """Flush any buffered audio by sending a flush event to Fish Audio."""
        logger.trace(f"{self}: Flushing audio buffers")
        if not self._websocket or self._websocket.state is State.CLOSED:
            return
        flush_message = {"event": "flush"}
        await self._get_websocket().send(ormsgpack.packb(flush_message))

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Stop all metrics and clean up retry state when audio context is interrupted."""
        self._pending_texts.pop(context_id, None)
        self._retry_counts.pop(context_id, None)
        await self.stop_all_metrics()
        await super().on_audio_context_interrupted(context_id)

    def _schedule_reconnect_after_error(self, context_id: str | None = None):
        if self._reconnect_task and not self._reconnect_task.done():
            logger.debug(f"{self}: [INTERNAL_RETRY] reconnect already scheduled")
            return
        logger.info(f"{self}: [INTERNAL_RETRY] scheduling WS reconnect context={context_id}")
        self._receive_task = None
        # Signal to the parent _receive_task_handler that we are handling
        # reconnection ourselves so it skips its own redundant reconnect.
        self._reconnect_in_progress = True
        self._reconnect_event.clear()
        self._reconnect_task = self.create_task(
            self._reconnect_and_resend(context_id), name="fish_reconnect"
        )

    async def _reconnect_and_resend(self, context_id: str | None = None):
        try:
            # Keep audio context alive during reconnect so the serialization
            # queue doesn't move on to the next context.
            if context_id and self.audio_context_available(context_id):
                self._refresh_audio_context(context_id)

            await self._disconnect_websocket()
            await self._connect_websocket()

            if not self._websocket:
                logger.error(f"{self}: [INTERNAL_RETRY] reconnect failed, WS not available")
                if context_id:
                    await self._exhaust_and_propagate_error(context_id)
                return

            # Keep context alive after reconnect
            if context_id and self.audio_context_available(context_id):
                self._refresh_audio_context(context_id)

            # Resend all pending texts for this context in order
            pending = self._pending_texts.get(context_id, []) if context_id else []
            for text in pending:
                logger.info(
                    f"{self}: [INTERNAL_RETRY] resending text context={context_id} "
                    f"text={text[:80]!r}"
                )
                await self._websocket.send(ormsgpack.packb({"event": "text", "text": text}))
                await self._websocket.send(ormsgpack.packb({"event": "flush"}))

            # Start new receive loop
            if not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error)
                )
        except Exception as e:
            logger.error(f"{self}: [INTERNAL_RETRY] reconnect failed: {e}")
            if context_id:
                await self._exhaust_and_propagate_error(context_id, exception=e)
        finally:
            self._reconnect_task = None
            self._reconnect_in_progress = False
            self._reconnect_event.set()

    async def _exhaust_and_propagate_error(
        self, context_id: str, exception: Exception | None = None
    ):
        """Push ErrorFrame and close the audio context after internal retries exhausted."""
        pending = self._pending_texts.pop(context_id, [])
        self._retry_counts.pop(context_id, None)
        texts_summary = "; ".join(t[:80] for t in pending) if pending else "(none)"
        logger.error(
            f"{self}: [INTERNAL_RETRY] exhausted {self._max_internal_retries} retries "
            f"for context={context_id}, failing over. pending_texts=[{texts_summary}]"
        )
        await self.push_error(
            error_msg=(
                f"Fish Audio synthesis failed after {self._max_internal_retries} "
                f"internal retries (context={context_id})"
            ),
            exception=exception,
        )
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def _receive_messages(self):
        import time

        # Gap detection: Fish Audio returns one audio chunk per flush.
        # A gap > threshold between chunks indicates a sentence boundary.
        _GAP_MIN_MS = 200  # below this, not a sentence boundary
        _GAP_MAX_MS = 5000  # above this, likely a turn boundary (skip silence)
        _last_audio_time = 0.0

        # Diagnostics: track received audio for error reporting
        _audio_chunks_received = 0
        _audio_bytes_received = 0
        _audio_duration_s = 0.0

        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    msg = ormsgpack.unpackb(message)
                    if isinstance(msg, dict):
                        event = msg.get("event")
                        if event == "audio":
                            audio_data = msg.get("audio")
                            # Only process larger chunks to remove msgpack overhead
                            if audio_data and len(audio_data) > 1024:
                                context_id = self.get_active_audio_context_id()
                                retry_group_id = self._retry_group_for_context(context_id)
                                logger.debug(
                                    f"{self}: recv Fish audio event context={context_id} "
                                    f"retry_group={retry_group_id or context_id} "
                                    f"bytes={len(audio_data)}"
                                )

                                # Detect sentence boundary and insert silence
                                if (
                                    self._inter_utterance_silence_s > 0
                                    and _last_audio_time > 0
                                    and context_id
                                    and self.audio_context_available(context_id)
                                ):
                                    gap_ms = (time.monotonic() - _last_audio_time) * 1000
                                    if _GAP_MIN_MS < gap_ms < _GAP_MAX_MS:
                                        logger.info(
                                            f"{self}: sentence boundary detected "
                                            f"context={context_id} "
                                            f"retry_group={retry_group_id or context_id} "
                                            f"gap_ms={gap_ms:.0f}"
                                        )
                                        boundary = TTSSentenceBoundaryFrame(
                                            context_id=context_id,
                                        )
                                        await self.append_to_audio_context(context_id, boundary)
                                        num_bytes = int(
                                            self._inter_utterance_silence_s * self.sample_rate * 2
                                        )
                                        silence = TTSAudioRawFrame(
                                            audio=b"\x00" * num_bytes,
                                            sample_rate=self.sample_rate,
                                            num_channels=1,
                                        )
                                        silence.metadata["_tts_silence"] = True
                                        await self.append_to_audio_context(context_id, silence)

                                frame = TTSAudioRawFrame(
                                    audio_data,
                                    self.sample_rate,
                                    1,
                                    context_id=context_id,
                                )
                                await self.append_to_audio_context(context_id, frame)
                                await self.stop_ttfb_metrics()
                                _last_audio_time = time.monotonic()
                                _audio_chunks_received += 1
                                _audio_bytes_received += len(audio_data)
                                _audio_duration_s += len(audio_data) / 2 / self.sample_rate
                        elif event == "finish":
                            reason = msg.get("reason", "unknown")
                            context_id = self.get_active_audio_context_id()
                            logger.info(
                                f"{self}: recv Fish finish event reason={reason} "
                                f"context={context_id} "
                                f"retry_group="
                                f"{self._retry_group_for_context(context_id) or context_id}"
                            )
                            if reason == "error":
                                count = (
                                    self._retry_counts.get(context_id, 0) + 1 if context_id else 1
                                )
                                if context_id:
                                    self._retry_counts[context_id] = count
                                # Diagnostic info
                                ctx_queue_size = 0
                                if context_id and self.audio_context_available(context_id):
                                    ctx_queue_size = self._audio_contexts[context_id].qsize()
                                logger.warning(
                                    f"{self}: [INTERNAL_RETRY] Fish finish reason=error "
                                    f"attempt={count}/{self._max_internal_retries} "
                                    f"context={context_id} | "
                                    f"audio_received: {_audio_chunks_received} chunks, "
                                    f"{_audio_bytes_received} bytes, "
                                    f"{_audio_duration_s:.1f}s | "
                                    f"audio_context_queue_size={ctx_queue_size}"
                                )
                                if count <= self._max_internal_retries:
                                    self._schedule_reconnect_after_error(context_id)
                                    return
                                else:
                                    if context_id:
                                        await self._exhaust_and_propagate_error(context_id)
                                    self._schedule_reconnect_after_error()
                                    return
                            else:
                                # Success — pop the first pending text
                                if context_id and context_id in self._pending_texts:
                                    pending = self._pending_texts[context_id]
                                    if pending:
                                        pending.pop(0)
                                    if not pending:
                                        del self._pending_texts[context_id]
                                if context_id:
                                    self._retry_counts.pop(context_id, None)
                                logger.debug(f"Fish Audio session finished: {reason}")

            except Exception as e:
                context_id = self.get_active_audio_context_id()
                logger.error(
                    f"{self}: [INTERNAL_RETRY] receive-loop exception "
                    f"context={context_id} error={e!r}"
                )
                if context_id:
                    await self._exhaust_and_propagate_error(context_id, exception=e)
                else:
                    await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
                self._schedule_reconnect_after_error()
                return

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Fish Audio's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames and control frames for the synthesized speech.
        """
        logger.debug(f"{self}: Generating Fish TTS: [{text}]")
        logger.info(
            f"{self}: run_tts context={context_id} "
            f"retry_group={self._retry_group_for_context(context_id) or context_id} "
            f"text={text[:160]!r}"
        )

        # Wait if a reconnect is in progress (from a previous sentence's error)
        await self._reconnect_event.wait()

        # Track this text as pending for internal retry
        self._pending_texts.setdefault(context_id, []).append(text)

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            # Send the text
            text_message = {
                "event": "text",
                "text": text,
            }
            try:
                logger.info(
                    f"{self}: send Fish text event context={context_id} "
                    f"retry_group={self._retry_group_for_context(context_id) or context_id}"
                )
                await self._get_websocket().send(ormsgpack.packb(text_message))
                await self.start_tts_usage_metrics(text)

                # Send flush event to force audio generation
                flush_message = {"event": "flush"}
                logger.info(
                    f"{self}: send Fish flush event context={context_id} "
                    f"retry_group={self._retry_group_for_context(context_id) or context_id}"
                )
                await self._get_websocket().send(ormsgpack.packb(flush_message))
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
