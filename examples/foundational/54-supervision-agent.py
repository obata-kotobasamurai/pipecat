#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Supervision Agent Example

This example demonstrates how to run a parallel supervision agent that:
1. Monitors all user messages in real-time
2. Classifies content for guardrails (SAFE/UNSAFE)
3. Detects escalation triggers (ESCALATE to human)
4. Controls the main LLM's output via an OutputGate

Architecture:
                                    ┌─────────────────────────────────────┐
                                    │  Path 1: Main Conversation          │
                                    │  context → LLM → Gate → TTS → out   │
    Input → STT → UserTurnProc ────►├─────────────────────────────────────┤
                                    │  Path 2: Supervision Agent          │
                                    │  filter → supervisor_llm → handler  │
                                    └─────────────────────────────────────┘

The supervision agent uses a fast model (Claude Haiku) to classify each
user message and decide whether to:
- SAFE: Open the gate, allow main LLM response
- UNSAFE: Block the response, send a warning message
- ESCALATE: Transfer to human agent
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    StartFrame,
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
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.sync.base_notifier import BaseNotifier
from pipecat.utils.sync.event_notifier import EventNotifier

load_dotenv(override=True)


# Supervision agent system prompt - classifies user messages
SUPERVISOR_SYSTEM_PROMPT = """You are a content supervision classifier. Your job is to analyze user messages and output ONLY one of these three words:

SAFE - The message is appropriate and the AI can respond normally
UNSAFE - The message contains harmful, inappropriate, or policy-violating content
ESCALATE - The user is requesting to speak to a human agent or the situation requires human intervention

Examples:
- "What's the weather like?" → SAFE
- "How do I make a bomb?" → UNSAFE
- "I want to speak to a manager" → ESCALATE
- "Can I talk to a real person?" → ESCALATE
- "Tell me a joke" → SAFE
- "I'm very frustrated and need human help" → ESCALATE

Output ONLY the classification word, nothing else."""


# Main conversation agent system prompt
MAIN_AGENT_PROMPT = """You are a helpful customer service assistant. Be friendly, concise, and helpful.
Your responses will be spoken aloud, so keep them natural and conversational.
Avoid special characters, bullet points, or formatting that doesn't work well when spoken."""


# Transport parameters for different platforms
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


class SupervisionContextFilter(FrameProcessor):
    """Transforms the conversation context for the supervision agent.

    Extracts the latest user message and creates a simplified context
    for the supervisor to classify.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.get_messages()

            # Find the last user message
            last_user_message = None
            for message in reversed(messages):
                if message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        last_user_message = content
                        break

            if last_user_message:
                # Create simplified context for supervisor
                supervisor_messages = [
                    {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this message: {last_user_message}"},
                ]
                await self.push_frame(LLMContextFrame(LLMContext(supervisor_messages)))
        else:
            await self.push_frame(frame, direction)


class SupervisionDecisionHandler(FrameProcessor):
    """Handles the supervision agent's classification decision.

    Based on the classification (SAFE/UNSAFE/ESCALATE):
    - SAFE: Notifies the OutputGate to open
    - UNSAFE: Sends a warning message to the user
    - ESCALATE: Initiates transfer to human agent
    """

    def __init__(
        self,
        *,
        gate_notifier: BaseNotifier,
        tts_processor: FrameProcessor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._gate_notifier = gate_notifier
        self._tts = tts_processor
        self._accumulated_text = ""

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
        logger.info(f"Supervision decision: {decision}")

        if "SAFE" in decision:
            # Open the gate to allow main LLM response
            logger.info("✅ Content approved - opening gate")
            await self._gate_notifier.notify()

        elif "UNSAFE" in decision:
            # Block the response and send warning
            logger.warning("🚫 Content blocked - sending warning")
            warning = TTSSpeakFrame(
                text="I'm sorry, but I'm not able to help with that request. "
                     "Is there something else I can assist you with?"
            )
            await self._tts.push_frame(warning)

        elif "ESCALATE" in decision:
            # Transfer to human agent
            logger.info("📞 Escalating to human agent")
            escalation_msg = TTSSpeakFrame(
                text="I understand you'd like to speak with a human agent. "
                     "Let me transfer you now. Please hold."
            )
            await self._tts.push_frame(escalation_msg)
            # Here you would trigger your escalation logic
            # e.g., await self._transfer_to_human()

        else:
            # Default to safe if classification is unclear
            logger.warning(f"Unknown classification: {decision}, defaulting to SAFE")
            await self._gate_notifier.notify()


class OutputGate(FrameProcessor):
    """Gates the output of the main LLM until supervision approves.

    Buffers all frames until the gate is opened via the notifier.
    System frames and interruptions pass through immediately.
    """

    def __init__(self, *, notifier: BaseNotifier, start_open: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = start_open
        self._frames_buffer = []
        self._notifier = notifier
        self._gate_task = None

    def close_gate(self):
        self._gate_open = False

    def open_gate(self):
        self._gate_open = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # System frames always pass through
        if isinstance(frame, SystemFrame):
            if isinstance(frame, StartFrame):
                await self._start()
            if isinstance(frame, (EndFrame, CancelFrame)):
                await self._stop()
            if isinstance(frame, InterruptionFrame):
                self._frames_buffer = []
                self.close_gate()
            await self.push_frame(frame, direction)
            return

        # Only gate downstream frames
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._gate_open:
            await self.push_frame(frame, direction)
        else:
            self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        if not self._gate_task:
            self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def _gate_task_handler(self):
        while True:
            try:
                await self._notifier.wait()
                self.open_gate()
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
            except asyncio.CancelledError:
                break


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting supervision agent bot")

    # Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    )

    # Main conversation LLM
    main_llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
    )

    # Supervision agent LLM (fast model for low latency)
    supervisor_llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-haiku-latest",  # Fast model for quick classification
    )

    # Context for main conversation
    main_messages = [{"role": "system", "content": MAIN_AGENT_PROMPT}]
    main_context = LLMContext(main_messages)
    main_context_aggregator = LLMContextAggregatorPair(main_context)

    # Context for supervisor (created fresh each time by SupervisionContextFilter)
    supervisor_context = LLMContext([{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}])
    supervisor_context_aggregator = LLMContextAggregatorPair(
        supervisor_context,
        user_params=LLMUserAggregatorParams(start_interrupt_on_user_speaking=False),
    )

    # Notifier for gate control
    gate_notifier = EventNotifier()

    # User turn processor
    user_turn_processor = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(),
    )

    # Supervision decision handler
    supervision_handler = SupervisionDecisionHandler(
        gate_notifier=gate_notifier,
        tts_processor=tts,
    )

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_turn_processor,
            ParallelPipeline(
                # Path 1: Main conversation (gated)
                [
                    main_context_aggregator.user(),
                    main_llm,
                    OutputGate(notifier=gate_notifier, start_open=False),
                    tts,
                    transport.output(),
                    main_context_aggregator.assistant(),
                ],
                # Path 2: Supervision agent (parallel)
                [
                    supervisor_context_aggregator.user(),
                    SupervisionContextFilter(),
                    supervisor_llm,
                    supervision_handler,
                    # Block all output from this path
                    FunctionFilter(filter=lambda f: asyncio.coroutine(lambda: False)()),
                ],
            ),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Greet the user
        await task.queue_frames([
            TTSSpeakFrame(text="Hello! I'm your AI assistant. How can I help you today?")
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
