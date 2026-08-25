"""OpenAI Embeddings API 的真实客户端适配器。"""

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .embedding_client import (
    EmbeddingBatch,
    EmbeddingClientError,
    EmbeddingVector,
    validate_embedding_texts,
    validate_embedding_vector,
)


class UnsupportedEmbeddingProviderError(ValueError):
    """配置的 Embedding 供应商不属于当前已审核范围。"""


EmbeddingProviderErrorCode = Literal[
    "embedding_timeout",
    "embedding_connection_error",
    "embedding_rate_limited",
    "embedding_server_error",
    "embedding_request_rejected",
    "embedding_provider_error",
]


class EmbeddingProviderRequestError(EmbeddingClientError):
    """供应商请求失败，且不向上层暴露 SDK 内部详情。"""

    def __init__(
        self,
        *,
        code: EmbeddingProviderErrorCode,
        retryable: bool,
    ) -> None:
        super().__init__("Embedding 供应商请求失败")
        self.code = code
        self.retryable = retryable


class InvalidEmbeddingProviderResponseError(RuntimeError):
    """供应商响应无法转换为统一 Embedding 契约。"""


class EmbeddingSettings(BaseSettings):
    """从环境读取真实 Embedding 的供应商、模型版本和密钥。"""

    model_config = SettingsConfigDict(
        env_prefix="FILENEST_EMBEDDING_",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: SecretStr

    @field_validator("provider", "model")
    @classmethod
    def reject_invalid_identifier(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("must be non-empty without surrounding whitespace")
        return value

    @field_validator("api_key")
    @classmethod
    def reject_invalid_api_key(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if not secret or secret != secret.strip():
            raise ValueError("must be non-empty without surrounding whitespace")
        return value


class _EmbeddingSdkClient(Protocol):
    embeddings: Any


SUPPORTED_PROVIDER_BASE_URLS = MappingProxyType({"openai": None})


class OpenAIEmbeddingClient:
    """将 OpenAI Embeddings API 响应转换为 FileNest 向量契约。"""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        sdk_client: _EmbeddingSdkClient | None = None,
    ) -> None:
        provider = settings.provider.casefold()
        if provider not in SUPPORTED_PROVIDER_BASE_URLS:
            raise UnsupportedEmbeddingProviderError(
                f"不支持的 Embedding 供应商: {settings.provider}"
            )

        self._model = settings.model
        self._last_model_version: str | None = None
        self._client = sdk_client or _build_sdk_client(settings, provider)

    @property
    def model_version(self) -> str:
        """返回最近一次响应确认的模型版本，未调用时返回请求模型。"""

        return self._last_model_version or self._model

    def embed(self, *, texts: Sequence[str]) -> EmbeddingBatch:
        """批量调用真实 Embedding API，并按输入顺序返回向量。"""

        normalized_texts = validate_embedding_texts(texts)
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=list(normalized_texts),
                encoding_format="float",
            )
        except APITimeoutError:
            raise EmbeddingProviderRequestError(
                code="embedding_timeout",
                retryable=True,
            ) from None
        except APIConnectionError:
            raise EmbeddingProviderRequestError(
                code="embedding_connection_error",
                retryable=True,
            ) from None
        except RateLimitError:
            raise EmbeddingProviderRequestError(
                code="embedding_rate_limited",
                retryable=True,
            ) from None
        except APIStatusError as error:
            is_server_error = error.status_code >= 500
            raise EmbeddingProviderRequestError(
                code=(
                    "embedding_server_error"
                    if is_server_error
                    else "embedding_request_rejected"
                ),
                retryable=is_server_error,
            ) from None
        except OpenAIError:
            raise EmbeddingProviderRequestError(
                code="embedding_provider_error",
                retryable=False,
            ) from None

        return _embedding_batch_from_response(
            response,
            expected_count=len(normalized_texts),
            requested_model=self._model,
            update_model_version=self._set_model_version,
        )

    def _set_model_version(self, model_version: str) -> None:
        self._last_model_version = model_version


def _build_sdk_client(
    settings: EmbeddingSettings,
    provider: str,
) -> _EmbeddingSdkClient:
    api_key = settings.api_key.get_secret_value()
    return cast(
        _EmbeddingSdkClient,
        OpenAI(
            api_key=api_key,
            base_url=SUPPORTED_PROVIDER_BASE_URLS[provider],
            max_retries=0,
            timeout=30.0,
        ),
    )


def _embedding_batch_from_response(
    response: object,
    *,
    expected_count: int,
    requested_model: str,
    update_model_version: Any,
) -> EmbeddingBatch:
    try:
        model_version = getattr(response, "model")
        data = getattr(response, "data")
        if (
            not isinstance(model_version, str)
            or not model_version.strip()
            or not isinstance(data, Sequence)
            or isinstance(data, (str, bytes))
            or len(data) != expected_count
        ):
            raise InvalidEmbeddingProviderResponseError(
                "Embedding 供应商响应不符合预期结构"
            )

        indexed_vectors: dict[int, EmbeddingVector] = {}
        for item in data:
            index = getattr(item, "index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= expected_count
                or index in indexed_vectors
            ):
                raise InvalidEmbeddingProviderResponseError(
                    "Embedding 供应商响应索引无效"
                )
            indexed_vectors[index] = validate_embedding_vector(
                getattr(item, "embedding")
            )

        if set(indexed_vectors) != set(range(expected_count)):
            raise InvalidEmbeddingProviderResponseError(
                "Embedding 供应商响应缺少向量"
            )
    except InvalidEmbeddingProviderResponseError:
        raise
    except (AttributeError, TypeError, ValueError, EmbeddingClientError):
        raise InvalidEmbeddingProviderResponseError(
            "Embedding 供应商响应不符合预期结构"
        ) from None

    if not model_version.strip():
        raise InvalidEmbeddingProviderResponseError(
            "Embedding 供应商响应不符合预期结构"
        )
    if not requested_model.strip():
        raise InvalidEmbeddingProviderResponseError(
            "Embedding 请求模型不能为空"
        )
    update_model_version(model_version)
    return tuple(indexed_vectors[index] for index in range(expected_count))
