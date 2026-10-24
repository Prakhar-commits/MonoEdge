"""Unit tests for OpenAICompatibleLLM using a fake SDK client (no network)."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.services.llm import OpenAICompatibleLLM


class FakeCompletions:
    """Returns (or raises) the scripted outcomes in order and records every call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def reply(content="hello", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def api_error(cls, message):
    response = httpx.Response(400, request=httpx.Request("POST", "http://llm.test"))
    return cls(message, response=response, body=None)


def make_llm(outcomes):
    completions = FakeCompletions(outcomes)
    llm = OpenAICompatibleLLM("http://llm.test", "key", "some-model")
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


MESSAGES = [{"role": "user", "content": "hi"}]


def test_sends_temperature_zero_and_returns_text():
    llm, completions = make_llm([reply("pong")])
    assert llm.complete(MESSAGES).text == "pong"
    assert completions.calls[0]["temperature"] == 0
    assert completions.calls[0]["model"] == "some-model"
    assert "tools" not in completions.calls[0]


def test_drops_temperature_when_model_rejects_it_and_remembers():
    rejected = api_error(openai.BadRequestError, "`temperature` is deprecated for this model.")
    llm, completions = make_llm([rejected, reply("one"), reply("two")])

    assert llm.complete(MESSAGES).text == "one"
    assert "temperature" in completions.calls[0]
    assert "temperature" not in completions.calls[1]  # the retry

    assert llm.complete(MESSAGES).text == "two"
    assert len(completions.calls) == 3  # no wasted attempt the second time
    assert "temperature" not in completions.calls[2]


def test_other_bad_requests_are_not_swallowed():
    llm, completions = make_llm([api_error(openai.BadRequestError, "model does not exist")])
    with pytest.raises(openai.BadRequestError):
        llm.complete(MESSAGES)
    assert len(completions.calls) == 1


def test_rate_limit_is_retried_once(monkeypatch):
    monkeypatch.setattr("app.services.llm.time.sleep", lambda seconds: None)
    limited = api_error(openai.RateLimitError, "slow down")
    llm, completions = make_llm([limited, reply("ok")])
    assert llm.complete(MESSAGES).text == "ok"
    assert len(completions.calls) == 2


def test_second_rate_limit_is_raised(monkeypatch):
    monkeypatch.setattr("app.services.llm.time.sleep", lambda seconds: None)
    limited = api_error(openai.RateLimitError, "slow down")
    llm, completions = make_llm([limited, limited])
    with pytest.raises(openai.RateLimitError):
        llm.complete(MESSAGES)
    assert len(completions.calls) == 2  # exactly one retry, not a loop


def test_tool_calls_are_parsed_and_tools_are_forwarded():
    call = SimpleNamespace(function=SimpleNamespace(name="create_ticket", arguments='{"a": 1}'))
    llm, completions = make_llm([reply(content=None, tool_calls=[call])])
    tools = [{"type": "function", "function": {"name": "create_ticket"}}]

    response = llm.complete(MESSAGES, tools=tools)

    assert completions.calls[0]["tools"] == tools
    assert response.text == ""  # a tool-only reply has no text
    assert [(c.name, c.arguments) for c in response.tool_calls] == [("create_ticket", '{"a": 1}')]


def test_empty_api_key_uses_a_placeholder(monkeypatch):
    captured = {}
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: captured.update(kwargs) or "client")
    llm = OpenAICompatibleLLM("http://localhost:11434/v1", "", "llama3")
    assert llm._get_client() == "client"
    assert captured["api_key"] == "not-needed"  # the SDK refuses an empty key
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 0
