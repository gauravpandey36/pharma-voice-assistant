"""
LLM abstraction layer supporting multiple providers.

Supports Ollama (local), Claude (Anthropic), and OpenAI APIs.
Follows the Gourav Digital Twin persona for pharmaceutical expertise.

Author: Gourav Pandey
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DIGITAL_TWIN_SYSTEM_PROMPT = (
    "You are the Digital Twin of Gourav Pandey, a pharmaceutical quality expert "
    "with deep expertise in GMP compliance, regulatory affairs, quality systems, "
    "and AI applications in regulated life sciences. You speak with authority on "
    "pharmaceutical manufacturing, quality control, validation, and regulatory "
    "requirements. You cite specific regulations (21 CFR, EU GMP, ICH guidelines) "
    "when relevant. Your responses are precise, actionable, and grounded in "
    "pharmaceutical domain knowledge. You communicate clearly and professionally."
)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    ttft_s: float  # Time to first token

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 3),
            "ttft_s": round(self.ttft_s, 3),
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: str = "",
        system_prompt: str = DIGITAL_TWIN_SYSTEM_PROMPT,
    ) -> LLMResponse:
        """Generate a response to a query."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        ...


class OllamaProvider(LLMProvider):
    """Local LLM via Ollama."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ) -> None:
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(
        self,
        query: str,
        context: str = "",
        system_prompt: str = DIGITAL_TWIN_SYSTEM_PROMPT,
    ) -> LLMResponse:
        """Generate response using local Ollama model."""
        import ollama

        client = ollama.Client(host=self._host)

        full_system = system_prompt
        if context:
            full_system += f"\n\nContext from knowledge base:\n{context}"

        start = time.time()
        first_token_time = None
        full_response = ""

        # Use streaming to measure TTFT
        stream = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.3},
            stream=True,
        )

        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
            full_response += chunk["message"]["content"]

        total_time = time.time() - start
        ttft = (first_token_time - start) if first_token_time else total_time

        # Estimate token counts
        est_input = len(full_system + query) // 4
        est_output = len(full_response) // 4

        return LLMResponse(
            text=full_response.strip(),
            model=self._model,
            provider="ollama",
            input_tokens=est_input,
            output_tokens=est_output,
            latency_s=total_time,
            ttft_s=ttft,
        )

    def is_available(self) -> bool:
        try:
            import ollama
            client = ollama.Client(host=self._host)
            client.list()
            return True
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "ollama"


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model

    def generate(
        self,
        query: str,
        context: str = "",
        system_prompt: str = DIGITAL_TWIN_SYSTEM_PROMPT,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        import openai

        client = openai.OpenAI(api_key=self._api_key)

        full_system = system_prompt
        if context:
            full_system += f"\n\nContext from knowledge base:\n{context}"

        start = time.time()
        first_token_time = None
        full_response = ""

        stream = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            stream=True,
        )

        for chunk in stream:
            if first_token_time is None and chunk.choices[0].delta.content:
                first_token_time = time.time()
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        total_time = time.time() - start
        ttft = (first_token_time - start) if first_token_time else total_time

        est_input = len(full_system + query) // 4
        est_output = len(full_response) // 4

        return LLMResponse(
            text=full_response.strip(),
            model=self._model,
            provider="openai",
            input_tokens=est_input,
            output_tokens=est_output,
            latency_s=total_time,
            ttft_s=ttft,
        )

    def is_available(self) -> bool:
        return bool(self._api_key)

    @property
    def provider_name(self) -> str:
        return "openai"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model

    def generate(
        self,
        query: str,
        context: str = "",
        system_prompt: str = DIGITAL_TWIN_SYSTEM_PROMPT,
    ) -> LLMResponse:
        """Generate response using Anthropic Claude API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)

        full_system = system_prompt
        if context:
            full_system += f"\n\nContext from knowledge base:\n{context}"

        start = time.time()
        first_token_time = None
        full_response = ""

        with client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=full_system,
            messages=[{"role": "user", "content": query}],
        ) as stream:
            for text in stream.text_stream:
                if first_token_time is None:
                    first_token_time = time.time()
                full_response += text

        total_time = time.time() - start
        ttft = (first_token_time - start) if first_token_time else total_time

        est_input = len(full_system + query) // 4
        est_output = len(full_response) // 4

        return LLMResponse(
            text=full_response.strip(),
            model=self._model,
            provider="anthropic",
            input_tokens=est_input,
            output_tokens=est_output,
            latency_s=total_time,
            ttft_s=ttft,
        )

    def is_available(self) -> bool:
        return bool(self._api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"


class LLMEngine:
    """
    Unified LLM engine with automatic provider selection and fallback.
    """

    _provider_classes: dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }

    def __init__(self, preferred_provider: str | None = None) -> None:
        """
        Initialize the LLM engine.

        Args:
            preferred_provider: Preferred provider name. Defaults to LLM_PROVIDER env var.
        """
        self._preferred = preferred_provider or os.environ.get("LLM_PROVIDER", "ollama")
        self._providers: dict[str, LLMProvider] = {}

        for name, cls in self._provider_classes.items():
            try:
                self._providers[name] = cls()
            except Exception as e:
                logger.debug("Provider '%s' not available: %s", name, e)

    def generate(
        self,
        query: str,
        context: str = "",
        provider: str | None = None,
    ) -> LLMResponse:
        """
        Generate a response using the best available provider.

        Args:
            query: User query.
            context: Optional RAG context.
            provider: Override provider selection.

        Returns:
            LLMResponse from the selected provider.

        Raises:
            RuntimeError: If no provider is available.
        """
        target = provider or self._preferred

        # Try preferred provider first
        if target in self._providers and self._providers[target].is_available():
            try:
                return self._providers[target].generate(query, context)
            except Exception as e:
                logger.warning("Provider '%s' failed: %s. Trying fallback.", target, e)

        # Fallback order
        fallback_order = ["ollama", "openai", "anthropic"]
        for name in fallback_order:
            if name == target:
                continue
            if name in self._providers and self._providers[name].is_available():
                try:
                    logger.info("Falling back to provider: %s", name)
                    return self._providers[name].generate(query, context)
                except Exception as e:
                    logger.warning("Fallback provider '%s' failed: %s", name, e)

        raise RuntimeError("No LLM provider available. Configure at least one provider in .env")

    def list_available(self) -> list[str]:
        """Return list of available provider names."""
        return [name for name, p in self._providers.items() if p.is_available()]
