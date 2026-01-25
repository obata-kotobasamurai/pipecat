#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Supervision Agent Example - Simple Pattern

This example demonstrates a simple parallel supervision agent that:
1. Monitors the conversation between user and assistant
2. If escalation criteria is met, emits an event for escalation

Architecture:
                                    ┌─────────────────────────────────────┐
                                    │  Path 1: Main Conversation          │
                                    │  context → LLM → TTS → output       │
    Input → STT → UserTurnProc ────►├─────────────────────────────────────┤
                                    │  Path 2: Supervision Agent          │
                                    │  observer → supervisor_llm → handler│
                                    └─────────────────────────────────────┘

The supervisor uses a fast model (Claude Haiku) to monitor each exchange
and emit events when action is needed:
- on_escalation_requested: User wants human agent
- on_unsafe_content: Content policy violation detected
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    SystemFrame,
    TextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv(override=True)


# Supervision agent system prompt
SUPERVISOR_SYSTEM_PROMPT = """You monitor conversations and output ONLY one word:

OK - Conversation is normal, no action needed
ESCALATE - User wants to speak to a human agent
UNSAFE - Content violates policies

Examples:
- User asks a normal question → OK
- "I want to speak to a manager" → ESCALATE
- "Can I talk to a real person?" → ESCALATE
- Harmful/inappropriate content → UNSAFE

Output ONLY: OK, ESCALATE, or UNSAFE"""


MAIN_AGENT_PROMPT = """You are a helpful customer service assistant. Be friendly and concise.
Your responses will be spoken aloud, so keep them natural and conversational."""


transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
}


class ConversationObserver(FrameProcessor):
    """Observes the conversation and creates context for the supervisor.

    Extracts the latest user message for the supervisor to evaluate.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.get_messages()

            # Get the last user message
            last_user_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        last_user_message = content
                        break

            if last_user_message:
                # Create supervisor context
                supervisor_messages = [
                    {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": last_user_message},
                ]
                await self.push_frame(LLMContextFrame(LLMContext(supervisor_messages)))
        else:
            await self.push_frame(frame, direction)


class SupervisionHandler(FrameProcessor):
    """Handles supervisor decisions and emits events.

    Emits:
    - on_escalation_requested: When user wants human agent
    - on_unsafe_content: When content policy is violated
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._accumulated_text = ""
        self._register_event_handler("on_escalation_requested")
        self._register_event_handler("on_unsafe_content")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self._accumulated_text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_decision()
            self._accumulated_text = ""
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)

    async def _handle_decision(self):
        decision = self._accumulated_text.strip().upper()
        logger.debug(f"Supervisor decision: {decision}")

        if "ESCALATE" in decision:
            logger.info("📞 Escalation requested")
            await self._call_event_handler("on_escalation_requested")

        elif "UNSAFE" in decision:
            logger.warning("🚫 Unsafe content detected")
            await self._call_event_handler("on_unsafe_content")

        # OK = no action needed


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot with supervision agent")

    # Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
    )

    # Main LLM
    main_llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
    )

    # Supervisor LLM (fast model)
    supervisor_llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-haiku-latest",
    )

    # Contexts
    main_context = LLMContext([{"role": "system", "content": MAIN_AGENT_PROMPT}])
    main_aggregator = LLMContextAggregatorPair(main_context)

    supervisor_context = LLMContext([{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}])
    supervisor_aggregator = LLMContextAggregatorPair(
        supervisor_context,
        user_params=LLMUserAggregatorParams(start_interrupt_on_user_speaking=False),
    )

    # User turn processor
    user_turn_processor = UserTurnProcessor(user_turn_strategies=UserTurnStrategies())

    # Supervision handler
    supervision_handler = SupervisionHandler()

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_turn_processor,
            ParallelPipeline(
                # Path 1: Main conversation
                [
                    main_aggregator.user(),
                    main_llm,
                    tts,
                    transport.output(),
                    main_aggregator.assistant(),
                ],
                # Path 2: Supervisor (monitors in parallel)
                [
                    supervisor_aggregator.user(),
                    ConversationObserver(),
                    supervisor_llm,
                    supervision_handler,
                    FunctionFilter(filter=lambda f: asyncio.sleep(0, result=False)),
                ],
            ),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # === EVENT HANDLERS FOR ESCALATION ===

    @supervision_handler.event_handler("on_escalation_requested")
    async def handle_escalation(handler):
        """Handle escalation to human agent."""
        logger.info("Handling escalation...")

        # Say goodbye message
        await task.queue_frames([
            TTSSpeakFrame(
                text="I understand you'd like to speak with a human agent. "
                "Let me transfer you now."
            )
        ])

        # For Daily transport with SIP, you can transfer the call:
        if isinstance(transport, DailyTransport):
            # Transfer to human agent phone number
            # await transport.sip_call_transfer({"sipUri": "sip:agent@example.com"})
            pass

        # Or end the call and notify your backend
        # await task.cancel()

    @supervision_handler.event_handler("on_unsafe_content")
    async def handle_unsafe(handler):
        """Handle unsafe content detection."""
        logger.warning("Unsafe content handled")
        await task.queue_frames([
            TTSSpeakFrame(text="I'm not able to help with that request.")
        ])

    # === TRANSPORT EVENT HANDLERS ===

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await task.queue_frames([
            TTSSpeakFrame(text="Hello! How can I help you today?")
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
