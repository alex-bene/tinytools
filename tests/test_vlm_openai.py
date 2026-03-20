from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

openai = pytest.importorskip("openai")
openai_module = pytest.importorskip("src.tinytools.vlm.openai")

OpenAIAPIModel = openai_module.OpenAIAPIModel


def _make_response(content: str = "ok", finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)]
    )


def _make_model(tmp_path: pytest.TempPathFactory) -> OpenAIAPIModel:
    return OpenAIAPIModel(
        model="gpt-4o-mini", cache_folder=tmp_path, max_retries=0, base_url="http://localhost:8000/v1", api_key="test"
    )


def test_single_forward_forwards_sampling_and_extra_body(tmp_path: pytest.TempPathFactory) -> None:
    model = _make_model(tmp_path)
    parse_mock = AsyncMock(return_value=_make_response(content="hello"))
    model.client.chat.completions.parse = parse_mock

    response = asyncio.run(
        model.single_forward(
            prompt="hello",
            ignore_cache=True,
            no_cache=True,
            temperature=0.25,
            top_p=0.6,
            presence_penalty=0.3,
            extra_body={"x": {"y": 1}},
        )
    )

    assert response == "hello"
    kwargs = parse_mock.await_args.kwargs
    assert kwargs["temperature"] == 0.25
    assert kwargs["top_p"] == 0.6
    assert kwargs["presence_penalty"] == 0.3
    assert kwargs["extra_body"] == {"x": {"y": 1}}


def test_completion_with_retries_default_omit_and_none_passthrough(tmp_path: pytest.TempPathFactory) -> None:
    model = _make_model(tmp_path)
    parse_mock = AsyncMock(return_value=_make_response(content="hello"))
    model.client.chat.completions.parse = parse_mock
    messages = [{"role": "user", "content": "hello"}]

    asyncio.run(model.completion_with_retries(messages=messages))
    default_kwargs = parse_mock.await_args.kwargs
    assert default_kwargs["temperature"] is openai.omit
    assert default_kwargs["top_p"] is openai.omit
    assert default_kwargs["presence_penalty"] is openai.omit
    assert default_kwargs["extra_body"] is None

    parse_mock.reset_mock()
    asyncio.run(model.completion_with_retries(messages=messages, extra_body=None))
    explicit_none_kwargs = parse_mock.await_args.kwargs
    assert explicit_none_kwargs["extra_body"] is None


def test_cache_hash_includes_sampling_values(tmp_path: pytest.TempPathFactory) -> None:
    model = _make_model(tmp_path)
    parse_mock = AsyncMock(return_value=_make_response(content="cached"))
    model.client.chat.completions.parse = parse_mock

    first = asyncio.run(model.single_forward(prompt="same prompt", temperature=0.1, no_cache=False, ignore_cache=False))
    second = asyncio.run(
        model.single_forward(prompt="same prompt", temperature=0.1, no_cache=False, ignore_cache=False)
    )
    third = asyncio.run(model.single_forward(prompt="same prompt", temperature=0.2, no_cache=False, ignore_cache=False))

    assert first == "cached"
    assert second == "cached"
    assert third == "cached"
    assert parse_mock.await_count == 2
