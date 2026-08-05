"""供应商无关的 Embedding 客户端协议与基础校验。"""

import math
from collections.abc import Sequence
from numbers import Real
from typing import Protocol, TypeAlias, runtime_checkable


EmbeddingVector: TypeAlias = tuple[float, ...]
EmbeddingBatch: TypeAlias = tuple[EmbeddingVector, ...]


class EmbeddingClientError(ValueError):
    """Embedding 请求或向量不符合统一契约。"""


class EmbeddingInputError(EmbeddingClientError):
    """Embedding 输入文本不符合契约。"""


class InvalidEmbeddingVectorError(EmbeddingClientError):
    """Embedding 向量不符合后续相似度计算的契约。"""


@runtime_checkable
class EmbeddingClient(Protocol):
    """所有真实或离线 Embedding 客户端都必须提供的最小同步接口。"""

    def embed(self, *, texts: Sequence[str]) -> EmbeddingBatch:
        """将一批文本转换为保持输入顺序的向量。"""

        ...


def validate_embedding_texts(
    texts: Sequence[str],
) -> tuple[str, ...]:
    """校验并冻结一批待向量化文本，不改变文本原始内容。"""

    if isinstance(texts, (str, bytes)) or not texts:
        raise EmbeddingInputError("Embedding 文本集合不能为空")

    normalized: list[str] = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError("Embedding 文本必须是非空字符串")
        normalized.append(text)
    return tuple(normalized)


def validate_embedding_vector(
    values: Sequence[float],
) -> EmbeddingVector:
    """将向量转换为有限浮点元组，并拒绝空向量或非法数值。"""

    if isinstance(values, (str, bytes)) or not values:
        raise InvalidEmbeddingVectorError(
            "Embedding 向量必须包含至少一个有限数值"
        )

    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidEmbeddingVectorError(
                "Embedding 向量只能包含实数"
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise InvalidEmbeddingVectorError(
                "Embedding 向量只能包含有限数值"
            )
        vector.append(numeric_value)
    return tuple(vector)
