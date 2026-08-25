from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

import backend.app.openai_embedding_client as client_module
from backend.app.embedding_client import EmbeddingInputError
from backend.app.openai_embedding_client import (
    EmbeddingProviderRequestError,
    EmbeddingSettings,
    InvalidEmbeddingProviderResponseError,
    OpenAIEmbeddingClient,
    UnsupportedEmbeddingProviderError,
)


class StubEmbeddings:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> object:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class StubSdkClient:
    def __init__(self, embeddings: StubEmbeddings) -> None:
        self.embeddings = embeddings


def _settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        provider="openai",
        model="text-embedding-3-small",
        api_key="secret-for-test",
    )


def test_client_sends_batch_and_restores_provider_order() -> None:
    embeddings = StubEmbeddings(
        SimpleNamespace(
            model="text-embedding-3-small-2025-01-01",
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ],
        )
    )
    client = OpenAIEmbeddingClient(
        _settings(),
        sdk_client=StubSdkClient(embeddings),
    )

    vectors = client.embed(texts=("第一个文本", "第二个文本"))

    assert vectors == ((1.0, 0.0), (0.0, 1.0))
    assert client.model_version == "text-embedding-3-small-2025-01-01"
    assert embeddings.requests == [
        {
            "model": "text-embedding-3-small",
            "input": ["第一个文本", "第二个文本"],
            "encoding_format": "float",
        }
    ]


def test_client_rejects_invalid_response_and_blank_input() -> None:
    client = OpenAIEmbeddingClient(
        _settings(),
        sdk_client=StubSdkClient(
            StubEmbeddings(
                SimpleNamespace(
                    model="text-embedding-3-small",
                    data=[SimpleNamespace(index=0, embedding=[])],
                )
            )
        ),
    )

    with pytest.raises(
        InvalidEmbeddingProviderResponseError,
        match="响应不符合预期结构",
    ):
        client.embed(texts=("无效响应",))

    with pytest.raises(EmbeddingInputError, match="非空字符串"):
        client.embed(texts=("   ",))


def test_client_hides_provider_details_and_rejects_unknown_provider() -> None:
    exposed_secret = "secret-that-must-stay-hidden"
    client = OpenAIEmbeddingClient(
        _settings(),
        sdk_client=StubSdkClient(
            StubEmbeddings(error=OpenAIError(exposed_secret))
        ),
    )

    with pytest.raises(EmbeddingProviderRequestError) as error_info:
        client.embed(texts=("需要向量化的文本",))

    assert error_info.value.code == "embedding_provider_error"
    assert not error_info.value.retryable
    assert exposed_secret not in str(error_info.value)

    with pytest.raises(
        UnsupportedEmbeddingProviderError,
        match="不支持的 Embedding 供应商",
    ):
        OpenAIEmbeddingClient(
            EmbeddingSettings(
                provider="unknown-provider",
                model="example-embedding",
                api_key="secret-for-test",
            ),
            sdk_client=StubSdkClient(StubEmbeddings()),
        )


def test_client_builds_openai_sdk_without_retrying_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_openai(**options: Any) -> StubSdkClient:
        captured.update(options)
        return StubSdkClient(StubEmbeddings())

    monkeypatch.setattr(client_module, "OpenAI", fake_openai)

    OpenAIEmbeddingClient(_settings())

    assert captured == {
        "api_key": "secret-for-test",
        "base_url": None,
        "max_retries": 0,
        "timeout": 30.0,
    }
