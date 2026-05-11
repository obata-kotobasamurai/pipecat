#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter LLM service implementation.

This module provides an OpenAI-compatible interface for interacting with OpenRouter's API,
extending the base OpenAI LLM service functionality.
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import NOT_GIVEN, _NotGiven, assert_given, is_given


class OpenRouterProviderLatencyThreshold(BaseModel):
    """Latency thresholds by percentile for provider filtering.

    Endpoints that don't meet these thresholds are deprioritized (moved to the
    end of the list) rather than excluded entirely.

    Args:
        p50: 50th percentile latency threshold in seconds.
        p75: 75th percentile latency threshold in seconds.
        p90: 90th percentile latency threshold in seconds.
        p99: 99th percentile latency threshold in seconds.
    """

    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p99: float | None = None


class OpenRouterSortConfig(BaseModel):
    """Advanced sort configuration for cross-model routing.

    Args:
        by: Sort criterion. Use ``"latency"`` to sort by lowest latency,
            or ``"price"`` to sort by lowest price.
        partition: Set to ``"none"`` to route to the lowest-latency endpoint
            across all models (useful with model fallbacks).
    """

    by: str = "latency"
    partition: str | None = None


class OpenRouterProviderPreferences(BaseModel):
    """OpenRouter provider routing preferences.

    Controls how OpenRouter selects and prioritizes providers for your request.
    See https://openrouter.ai/docs/guides/routing/provider-selection for details.

    Args:
        sort: Provider sort order. Use ``"latency"`` to disable load balancing
            and try providers in order of lowest latency first. Use ``"price"``
            to sort by price. Can also be an ``OpenRouterSortConfig`` for
            advanced cross-model sorting with ``partition``.
        preferred_max_latency: Maximum acceptable latency threshold. A float
            applies to the p50 percentile. Use ``OpenRouterProviderLatencyThreshold``
            for per-percentile control.
        allow: List of provider names to allow (e.g., ``["OpenAI", "Anthropic"]``).
        deny: List of provider names to deny.
        order: Ordered list of provider names to try in sequence.
        require_parameters: If True, only use providers that support all
            parameters in the request.
        quantizations: List of allowed quantization levels
            (e.g., ``["bf16", "fp8"]``).
    """

    sort: str | OpenRouterSortConfig | None = None
    preferred_max_latency: float | OpenRouterProviderLatencyThreshold | None = None
    allow: list[str] | None = None
    deny: list[str] | None = None
    order: list[str] | None = None
    require_parameters: bool | None = None
    quantizations: list[str] | None = None


@dataclass
class OpenRouterLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for OpenRouterLLMService.

    Args:
        provider: OpenRouter provider routing preferences for controlling
            provider selection, latency-based sorting, and filtering.
    """

    provider: OpenRouterProviderPreferences | None | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class OpenRouterLLMService(OpenAILLMService):
    """A service for interacting with OpenRouter's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to OpenRouter's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    Settings = OpenRouterLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the OpenRouter LLM service.

        Args:
            api_key: The API key for accessing OpenRouter's API. If None, will attempt
                to read from environment variables.
            model: The model identifier to use. Defaults to "openai/gpt-4o-2024-11-20".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenRouterLLMService.Settings(model=...)`` instead.

            base_url: The base URL for OpenRouter API. Defaults to "https://openrouter.ai/api/v1".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="openai/gpt-4o-2024-11-20", provider=None)

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create an OpenRouter API client.

        Args:
            api_key: The API key to use for authentication. If None, uses instance default.
            base_url: The base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the parent client creation method.

        Returns:
            The configured OpenRouter API client instance.
        """
        logger.debug(f"Creating OpenRouter client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)

    def build_chat_completion_params(
        self, params_from_context: OpenAILLMInvocationParams
    ) -> dict[str, Any]:
        """Builds chat parameters, handling model-specific constraints.

        Includes OpenRouter-specific provider preferences for routing control
        (latency-based sorting, provider filtering, etc.).

        Args:
            params_from_context: Parameters from the LLM context.

        Returns:
            Transformed parameters ready for the API call.
        """
        params = super().build_chat_completion_params(params_from_context)

        if is_given(self._settings.provider) and self._settings.provider is not None:
            params["provider"] = self._settings.provider.model_dump(exclude_none=True)

        model = assert_given(self._settings.model)
        if model is not None and "gemini" in model.lower():
            messages = params.get("messages", [])
            if not messages:
                return params
            transformed_messages = []
            system_message_seen = False
            for msg in messages:
                if msg.get("role") == "system":
                    if not system_message_seen:
                        transformed_messages.append(msg)
                        system_message_seen = True
                    else:
                        new_msg = msg.copy()
                        new_msg["role"] = "user"
                        transformed_messages.append(new_msg)
                else:
                    transformed_messages.append(msg)
            params["messages"] = transformed_messages

        return params
