#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.soniox.stt import END_TOKEN, SonioxSTTService, _language_from_tokens
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, messages, *, state=State.OPEN, send_side_effect=None):
        self._messages = messages
        self.state = state
        self.send = AsyncMock(side_effect=send_side_effect)
        self.close = AsyncMock()

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


@pytest.mark.asyncio
async def test_connect_failure_clears_stale_websocket_without_raising(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise RuntimeError("connection failed")

    monkeypatch.setattr("pipecat.services.soniox.stt.websocket_connect", fake_websocket_connect)

    service = SonioxSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket([], state=State.CLOSED)
    service.push_error = AsyncMock()

    await service._connect_websocket()

    assert service._websocket is None
    service.push_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_preconnect_configures_sample_rate_without_starting_tasks(monkeypatch):
    websocket = _FakeWebsocket([])
    connect = AsyncMock(return_value=websocket)
    monkeypatch.setattr("pipecat.services.soniox.stt.websocket_connect", connect)

    service = SonioxSTTService(api_key="test-key", sample_rate=8000)

    assert await service.preconnect_websocket() is True

    connect.assert_awaited_once()
    config = json.loads(websocket.send.await_args.args[0])
    assert config["sample_rate"] == 8000
    assert service._receive_task is None
    assert service._keepalive_task is None


@pytest.mark.asyncio
async def test_start_adopts_preconnected_websocket_once(monkeypatch):
    websocket = _FakeWebsocket([])
    connect = AsyncMock(return_value=websocket)
    monkeypatch.setattr("pipecat.services.soniox.stt.websocket_connect", connect)

    service = SonioxSTTService(api_key="test-key", sample_rate=16000)
    assert await service.preconnect_websocket() is True

    keepalive_starts = 0
    receive_gate = asyncio.Event()

    def fake_create_keepalive_task():
        nonlocal keepalive_starts
        keepalive_starts += 1

    async def fake_receive_handler(_report_error):
        await receive_gate.wait()

    monkeypatch.setattr(service, "_create_keepalive_task", fake_create_keepalive_task)
    monkeypatch.setattr(service, "_receive_task_handler", fake_receive_handler)
    monkeypatch.setattr(service, "create_task", asyncio.create_task)

    from pipecat.frames.frames import StartFrame

    await service.start(StartFrame(audio_in_sample_rate=16000))

    connect.assert_awaited_once()
    assert websocket.send.await_count == 1
    assert keepalive_starts == 1
    assert service._receive_task is not None
    assert service._preconnected_websocket is False

    service._receive_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await service._receive_task


@pytest.mark.asyncio
async def test_cancelled_preconnect_closes_acquired_websocket(monkeypatch):
    send_started = asyncio.Event()

    async def blocked_send(_message):
        send_started.set()
        await asyncio.Event().wait()

    websocket = _FakeWebsocket([], send_side_effect=blocked_send)
    monkeypatch.setattr(
        "pipecat.services.soniox.stt.websocket_connect", AsyncMock(return_value=websocket)
    )
    service = SonioxSTTService(api_key="test-key", sample_rate=8000)

    task = asyncio.create_task(service.preconnect_websocket())
    await send_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    websocket.close.assert_awaited_once()
    assert service._websocket is None
    assert service._preconnected_websocket is False


@pytest.mark.asyncio
async def test_failed_preconnect_closes_socket_and_can_retry(monkeypatch):
    failed_websocket = _FakeWebsocket([], send_side_effect=RuntimeError("config failed"))
    healthy_websocket = _FakeWebsocket([])
    connect = AsyncMock(side_effect=[failed_websocket, healthy_websocket])
    monkeypatch.setattr("pipecat.services.soniox.stt.websocket_connect", connect)
    service = SonioxSTTService(api_key="test-key", sample_rate=8000)
    service.push_error = AsyncMock()

    assert await service.preconnect_websocket() is False
    failed_websocket.close.assert_awaited_once()
    assert service._websocket is None
    service.push_error.assert_not_awaited()

    assert await service.preconnect_websocket() is True
    assert connect.await_count == 2
    assert service._websocket is healthy_websocket


@pytest.mark.asyncio
async def test_receive_messages_reports_tokenless_error(monkeypatch):
    service = SonioxSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket(
        [json.dumps({"error_code": 401, "error_message": "invalid api key", "finished": True})]
    )
    errors = []

    async def fake_push_error(*, error_msg, **_kwargs):
        errors.append(error_msg)

    monkeypatch.setattr(service, "push_error", fake_push_error)

    await service._receive_messages()

    assert errors == ["Error: 401 (_receive_messages) - invalid api key"]


def test_language_from_tokens_uses_single_recognized_language():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": " world", "language": "en"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


def test_language_from_tokens_uses_most_common_language():
    tokens = [
        {"text": "Ik", "language": "nl"},
        {"text": " zoek", "language": "nl"},
        {"text": " computer", "language": "en"},
    ]

    assert _language_from_tokens(tokens) == Language.NL


def test_language_from_tokens_skips_unknown_language():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": "!", "language": "klingon"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


def test_language_from_tokens_skips_missing_language():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": " wereld"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


def test_language_from_tokens_ignores_unknown_and_missing_languages():
    tokens = [
        {"text": "Hello", "language": "klingon"},
        {"text": " world"},
        {"text": "!"},
    ]

    assert _language_from_tokens(tokens) is None


def test_language_from_tokens_uses_first_language_on_tie():
    tokens = [
        {"text": "Hello", "language": "en"},
        {"text": " wereld", "language": "nl"},
    ]

    assert _language_from_tokens(tokens) == Language.EN


@pytest.mark.asyncio
async def test_receive_messages_sets_final_transcription_language(monkeypatch):
    service = SonioxSTTService(api_key="test-key")
    pushed_frames = []
    traced_transcriptions = []

    async def fake_push_frame(frame):
        pushed_frames.append(frame)

    async def fake_handle_transcription(transcript, is_final, language=None):
        traced_transcriptions.append((transcript, is_final, language))

    async def fake_stop_processing_metrics():
        pass

    messages = [
        json.dumps(
            {
                "tokens": [
                    {"text": "Ik", "is_final": True, "language": "nl"},
                    {"text": " zoek", "is_final": True, "language": "nl"},
                    {"text": " computer", "is_final": True, "language": "en"},
                    {"text": END_TOKEN, "is_final": True},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]

    service._websocket = _FakeWebsocket(messages)
    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_handle_transcription)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_processing_metrics)

    await service._receive_messages()

    final_frames = [frame for frame in pushed_frames if isinstance(frame, TranscriptionFrame)]
    assert len(final_frames) == 1
    assert final_frames[0].text == "Ik zoek computer"
    assert final_frames[0].language == Language.NL
    assert final_frames[0].finalized is True
    assert final_frames[0].result == [
        {"text": "Ik", "is_final": True, "language": "nl"},
        {"text": " zoek", "is_final": True, "language": "nl"},
        {"text": " computer", "is_final": True, "language": "en"},
    ]
    assert traced_transcriptions == [("Ik zoek computer", True, Language.NL)]


@pytest.mark.asyncio
async def test_receive_messages_allows_final_transcription_without_language(monkeypatch):
    service = SonioxSTTService(api_key="test-key")
    pushed_frames = []
    traced_transcriptions = []

    async def fake_push_frame(frame):
        pushed_frames.append(frame)

    async def fake_handle_transcription(transcript, is_final, language=None):
        traced_transcriptions.append((transcript, is_final, language))

    async def fake_stop_processing_metrics():
        pass

    messages = [
        json.dumps(
            {
                "tokens": [
                    {"text": "Tell", "is_final": True},
                    {"text": " me", "is_final": True},
                    {"text": " a", "is_final": True},
                    {"text": " joke.", "is_final": True},
                    {"text": END_TOKEN, "is_final": True},
                ]
            }
        ),
        json.dumps({"tokens": [], "finished": True}),
    ]

    service._websocket = _FakeWebsocket(messages)
    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "_handle_transcription", fake_handle_transcription)
    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_processing_metrics)

    await service._receive_messages()

    final_frames = [frame for frame in pushed_frames if isinstance(frame, TranscriptionFrame)]
    assert len(final_frames) == 1
    assert final_frames[0].text == "Tell me a joke."
    assert final_frames[0].language is None
    assert final_frames[0].finalized is True
    assert traced_transcriptions == [("Tell me a joke.", True, None)]
