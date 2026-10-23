import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: str  # raw JSON string from the model; validated by the caller


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


class OpenAICompatibleLLM:
    """Works with any OpenAI-compatible endpoint (Groq, Gemini, Ollama, OpenAI)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self._client = None
        self._send_temperature = True

    def _get_client(self):
        if self._client is None:
            import openai

            # Some servers (Ollama) need no key, but the SDK refuses an empty one.
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "not-needed",
                timeout=30,
                max_retries=0,  # we do our own single retry below
            )
        return self._client

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        import openai

        kwargs = {"model": self.model, "messages": messages}
        if self._send_temperature:
            kwargs["temperature"] = 0  # deterministic answers
        if tools:
            kwargs["tools"] = tools
        client = self._get_client()
        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except openai.BadRequestError as exc:
                # Some newer models (e.g. recent Claude models) reject `temperature`.
                if "temperature" not in str(exc):
                    raise
                logger.warning("Model rejects temperature; sending requests without it")
                self._send_temperature = False
                kwargs.pop("temperature", None)
                response = client.chat.completions.create(**kwargs)
        except openai.RateLimitError:
            logger.warning("LLM rate limited; retrying once")
            time.sleep(2)
            response = client.chat.completions.create(**kwargs)

        message = response.choices[0].message
        tool_calls = [
            ToolCall(name=call.function.name, arguments=call.function.arguments)
            for call in (message.tool_calls or [])
        ]
        return LLMResponse(text=message.content or "", tool_calls=tool_calls)
