"""用于检索测试的离线、可预测 Embedding 客户端。"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .embedding_client import (
    EmbeddingBatch,
    EmbeddingClientError,
    EmbeddingVector,
    InvalidEmbeddingVectorError,
    validate_embedding_texts,
    validate_embedding_vector,
)


class FakeEmbeddingTextNotConfiguredError(EmbeddingClientError):
    """Fake Embedding 没有为请求文本配置向量。"""


class FakeEmbeddingClient:
    """按精确文本映射返回向量，并记录不可变调用快照。"""

    def __init__(
        self,
        embeddings: Mapping[str, Sequence[float]],
    ) -> None:
        texts = validate_embedding_texts(tuple(embeddings.keys()))
        validated: dict[str, EmbeddingVector] = {}
        dimension: int | None = None

        for text in texts:
            vector = validate_embedding_vector(embeddings[text])
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise InvalidEmbeddingVectorError(
                    "Fake Embedding 的向量维度必须一致"
                )
            validated[text] = vector

        self._embeddings = MappingProxyType(validated)
        self._dimension = dimension
        self._calls: list[tuple[str, ...]] = []

    @property
    def calls(self) -> tuple[tuple[str, ...], ...]:
        """返回调用文本的只读快照。"""

        return tuple(self._calls)

    @property
    def dimension(self) -> int:
        """返回该替身配置的统一向量维度。"""

        # 空映射已在 validate_embedding_texts 中拒绝，因此这里必有维度。
        assert self._dimension is not None
        return self._dimension

    def embed(self, *, texts: Sequence[str]) -> EmbeddingBatch:
        """按输入顺序返回预先配置的向量，不访问网络或外部模型。"""

        requested_texts = validate_embedding_texts(texts)
        self._calls.append(requested_texts)

        missing_texts = tuple(
            text for text in requested_texts if text not in self._embeddings
        )
        if missing_texts:
            missing = ", ".join(repr(text) for text in missing_texts)
            raise FakeEmbeddingTextNotConfiguredError(
                f"Fake Embedding 未配置文本对应的向量: {missing}"
            )

        return tuple(self._embeddings[text] for text in requested_texts)
