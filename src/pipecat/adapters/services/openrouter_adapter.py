#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter LLM adapter for Pipecat.

This adapter extends the OpenAI adapter to handle system message conversion
for providers like Gemini that don't support multiple system messages in
conversation history.
"""

from typing import Any, Dict, List

from openai.types.chat import ChatCompletionMessageParam

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage, LLMSpecificMessage


class OpenRouterLLMAdapter(OpenAILLMAdapter):
    """OpenRouter-specific adapter for Pipecat.

    Extends OpenAILLMAdapter to handle system message conversion for providers
    that don't support multiple system messages inline in conversation history
    (e.g., Gemini models via OpenRouter).

    The adapter converts mid-conversation system messages to user messages,
    keeping only the first system message as an actual system message. This
    matches the behavior of the native Gemini adapter.
    """

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages with system message conversion applied for accurate logging.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages showing the actual converted format sent to the API.
        """
        return self._from_universal_context_messages(self.get_messages(context))

    def _from_universal_context_messages(
        self, messages: List[LLMContextMessage]
    ) -> List[ChatCompletionMessageParam]:
        """Convert universal context messages to OpenAI format with system message handling.

        Converts mid-conversation system messages to user messages, keeping only
        the first system message as an actual system message. This is necessary
        for providers like Gemini that don't support multiple system messages.

        Args:
            messages: List of universal context messages.

        Returns:
            List of OpenAI-formatted messages with system messages converted as needed.
        """
        result = []
        has_system_message = False

        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                # Extract the actual message content from LLMSpecificMessage
                result.append(message.message)
            else:
                # Check if this is a system message
                role = message.get("role") if isinstance(message, dict) else None

                if role == "system":
                    if has_system_message:
                        # Convert subsequent system messages to user messages
                        converted_message = dict(message)
                        converted_message["role"] = "user"
                        result.append(converted_message)
                    else:
                        # Keep the first system message as-is
                        has_system_message = True
                        result.append(message)
                else:
                    # Non-system messages pass through unchanged
                    result.append(message)

        return result
