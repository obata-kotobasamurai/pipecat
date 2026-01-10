#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AmiVoice speech-to-text service implementation.

This module provides a WebSocket-based STT service that integrates with
the AmiVoice Cloud Platform API for real-time speech recognition.
"""

import json
from enum import Enum
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use AmiVoice, you need to `pip install pipecat-ai[amivoice]`.")
    raise Exception(f"Missing module: {e}")


class OutputFormat(str, Enum):
    """Output format for transcription text.

    Attributes:
        WRITTEN: Output in written form with kanji (default).
        SPOKEN: Output in spoken form with hiragana reading.
    """

    WRITTEN = "written"
    SPOKEN = "spoken"


class AmiVoiceInputParams(BaseModel):
    """Configuration parameters for AmiVoice STT.

    Parameters:
        engine: Speech recognition engine name. Defaults to "-a-general" (general-purpose).
        result_updated_interval: Interval for interim results in milliseconds. Defaults to 300.
        output_format: Output text format (written/spoken). Defaults to WRITTEN.
    """

    engine: str = "-a-general"
    result_updated_interval: int = 500
    output_format: OutputFormat = OutputFormat.WRITTEN


def _audio_format_from_sample_rate(sample_rate: int) -> str:
    """Get AmiVoice audio format string from sample rate.

    Args:
        sample_rate: Audio sample rate in Hz.

    Returns:
        AmiVoice audio format string ("8K" or "16K").
    """
    if sample_rate <= 8000:
        return "8K"
    return "16K"


class AmiVoiceSTTService(WebsocketSTTService):
    """Speech-to-text service using AmiVoice WebSocket API.

    This service connects to AmiVoice Cloud Platform's WebSocket API for
    real-time transcription with support for Japanese language recognition,
    including both written (kanji) and spoken (hiragana) output formats.

    For complete API documentation, see:
    https://docs.amivoice.com/amivoice-api/manual/reference/websocket/
    """

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://acp-api.amivoice.com/v1/nolog/",
        sample_rate: Optional[int] = 16000,
        params: Optional[AmiVoiceInputParams] = None,
        **kwargs,
    ):
        """Initialize the AmiVoice STT service.

        Args:
            api_key: AmiVoice API key (APPKEY).
            url: AmiVoice WebSocket API URL. Use "wss://acp-api.amivoice.com/v1/"
                for logging enabled, or "wss://acp-api.amivoice.com/v1/nolog/" for no logging.
            sample_rate: Audio sample rate. Defaults to 16000 Hz.
            params: Additional configuration parameters.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._params = params or AmiVoiceInputParams()

        self._api_key = api_key
        self._url = url
        self.set_model_name(self._params.engine)

        self._receive_task = None
        self._session_active = False
        self._session_ending = False  # True while waiting for 'e' response

        # Audio buffer for capturing audio before VAD detection
        self._audio_buffer = bytearray()
        self._audio_buffer_max_size = 0  # Set in start()

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the AmiVoice STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        # 1 second of 16-bit mono audio at sample_rate
        self._audio_buffer_max_size = self.sample_rate * 2
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the AmiVoice STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._end_session()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the AmiVoice STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def start_metrics(self):
        """Start performance metrics collection."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle VAD events.

        Uses Pipecat VAD for speech detection. When user starts speaking,
        a new recognition session is started. When user stops speaking,
        the session is ended to get final results.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_metrics()
            # Start new session when user starts speaking
            if not self._session_active:
                await self._start_session()

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # End session to get final results
            if self._session_active:
                await self._end_session()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to AmiVoice using p command.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are handled via WebSocket events.
        """
        if self._session_active:
            if self._websocket and self._websocket.state is State.OPEN:
                # p command: 'p' prefix + binary audio data
                await self._websocket.send(b"p" + audio)
        else:
            # Buffer audio before VAD detection (keep max ~1 second)
            self._audio_buffer.extend(audio)
            if len(self._audio_buffer) > self._audio_buffer_max_size:
                excess = len(self._audio_buffer) - self._audio_buffer_max_size
                del self._audio_buffer[:excess]

        yield None

    async def _connect(self):
        """Connect to the AmiVoice WebSocket server."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the AmiVoice WebSocket server."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to AmiVoice."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to AmiVoice STT")
            self._websocket = await websocket_connect(self._url)
            await self._call_event_handler("on_connected")
            logger.debug("Connected to AmiVoice STT")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to AmiVoice: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close the WebSocket connection to AmiVoice."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from AmiVoice STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _start_session(self):
        """Start a new recognition session with s command."""
        # Don't start new session while previous one is still ending
        if self._session_ending:
            logger.debug("Waiting for previous session to end, skipping start")
            return

        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect_websocket()

        # Determine audio format from sample rate
        audio_format = _audio_format_from_sample_rate(self.sample_rate)

        # Build s command: s <audio_format> <engine> <parameters>
        s_command = (
            f"s {audio_format} {self._params.engine} "
            f"authorization={self._api_key} "
            f"resultUpdatedInterval={self._params.result_updated_interval}"
        )

        logger.debug(f"Starting AmiVoice session: {s_command[:50]}...")
        await self._websocket.send(s_command)
        self._session_active = True

        # Send buffered audio captured before VAD detection
        if self._audio_buffer:
            await self._websocket.send(b"p" + bytes(self._audio_buffer))
            self._audio_buffer.clear()

    async def _end_session(self):
        """End the current recognition session with e command."""
        if self._websocket and self._websocket.state is State.OPEN and self._session_active:
            logger.debug("Ending AmiVoice session")
            self._session_active = False  # Stop audio from being sent
            self._session_ending = True  # Prevent new session until 'e' response
            self._audio_buffer.clear()  # Clear buffer for next turn
            await self._websocket.send("e")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The WebSocket connection.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("WebSocket not connected")

    async def _receive_messages(self):
        """Receive and process WebSocket messages from AmiVoice.

        AmiVoice uses a text-based protocol where the first character
        indicates the message type:
        - 's': Session start response
        - 'p': Audio receive response
        - 'e': Session end response
        - 'S': Speech start event
        - 'E': Speech end event
        - 'C': Recognition start event
        - 'U': Interim result event
        - 'A'/'R': Final result event
        - 'G': Server info event
        """
        async for message in self._get_websocket():
            try:
                # Skip binary messages
                if isinstance(message, bytes):
                    continue

                if len(message) < 1:
                    continue

                event_type = message[0]
                payload = message[1:].strip() if len(message) > 1 else ""

                await self._process_event(event_type, payload)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse AmiVoice message: {e}")
            except Exception as e:
                logger.error(f"Error processing AmiVoice message: {e}")

    async def _process_event(self, event_type: str, payload: str):
        """Process an AmiVoice event.

        Args:
            event_type: Single character event type.
            payload: Event payload (may be empty or JSON).
        """
        if event_type == "U":
            # Interim result
            await self._handle_interim_result(payload)

        elif event_type in ("A", "R"):
            # Final result
            await self._handle_final_result(payload)

        elif event_type == "S":
            # Speech start detected by server
            logger.trace(f"AmiVoice speech start: {payload}")

        elif event_type == "E":
            # Speech end detected by server
            logger.trace(f"AmiVoice speech end: {payload}")

        elif event_type == "C":
            # Recognition processing started
            logger.trace("AmiVoice recognition started")

        elif event_type == "G":
            # Server info (can be ignored)
            logger.trace(f"AmiVoice server info: {payload}")

        elif event_type == "s":
            # Session start response
            if payload:
                # Error case: "s <error_message>"
                await self.push_error(error_msg=f"AmiVoice session error: {payload}")
                self._session_active = False
            else:
                logger.debug("AmiVoice session started successfully")

        elif event_type == "p":
            # Audio receive response
            if payload:
                await self.push_error(error_msg=f"AmiVoice audio error: {payload}")

        elif event_type == "e":
            # Session end response
            self._session_active = False
            self._session_ending = False  # Allow new session to start
            if payload:
                await self.push_error(error_msg=f"AmiVoice session end error: {payload}")
            else:
                logger.debug("AmiVoice session ended successfully")
                # If audio was buffered while session was ending, start new session
                if self._audio_buffer:
                    await self._start_session()

        else:
            logger.debug(f"Unknown AmiVoice event: {event_type}")

    def _extract_text(self, data: dict) -> str:
        """Extract text from result based on output format setting.

        Args:
            data: Result data containing tokens with written/spoken fields.

        Returns:
            Extracted text in the configured format.
        """
        if self._params.output_format == OutputFormat.SPOKEN:
            # Extract spoken (hiragana) from tokens
            results = data.get("results", [])
            if results:
                tokens = results[0].get("tokens", [])
                # Use 'spoken' field if available, fall back to 'written'
                # Filter out '_' (AmiVoice uses '_' as placeholder for punctuation)
                text = "".join(t.get("spoken", t.get("written", "")) for t in tokens)
                return text.replace("_", "")

        # Default: use 'text' field (written form with kanji)
        return data.get("text", "")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _handle_interim_result(self, payload: str):
        """Handle interim transcription result (U event).

        Args:
            payload: JSON string containing interim result.
        """
        if not payload:
            return

        try:
            data = json.loads(payload)
            text = self._extract_text(data)

            if text:
                await self.stop_ttfb_metrics()
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=text,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        result=data,
                    )
                )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse interim result: {payload[:100]}")

    async def _handle_final_result(self, payload: str):
        """Handle final transcription result (A/R event).

        Args:
            payload: JSON string containing final result.
        """
        if not payload:
            return

        try:
            data = json.loads(payload)
            text = self._extract_text(data)

            if text:
                await self.stop_ttfb_metrics()
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        result=data,
                    )
                )
                await self._handle_transcription(text, is_final=True)
                await self.stop_processing_metrics()
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse final result: {payload[:100]}")
