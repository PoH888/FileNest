import pytest

from backend.app.embedding_client import (
    EmbeddingClient,
    InvalidEmbeddingVectorError,
)
from backend.app.fake_embedding_client import (
    FakeEmbeddingClient,
    FakeEmbeddingTextNotConfiguredError,
)


def test_fake_embedding_is_replaceable_and_preserves_batch_order() -> None:
    client = FakeEmbeddingClient(
        {
            "审批": (1.0, 0.0),
            "批准": (0.0, 1.0),
        }
    )
    texts = ["批准", "审批"]
    embedding_client: EmbeddingClient = client

    vectors = embedding_client.embed(texts=texts)
    texts.append("后来添加的文本")

    assert isinstance(client, EmbeddingClient)
    assert vectors == ((0.0, 1.0), (1.0, 0.0))
    assert client.calls == (("批准", "审批"),)
    assert client.dimension == 2


def test_fake_embedding_rejects_unconfigured_text() -> None:
    client = FakeEmbeddingClient({"已知文本": (1.0,)})

    with pytest.raises(
        FakeEmbeddingTextNotConfiguredError,
        match="未配置文本对应的向量",
    ):
        client.embed(texts=["未知文本"])

    assert client.calls == (("未知文本",),)


def test_fake_embedding_rejects_inconsistent_vector_dimensions() -> None:
    with pytest.raises(
        InvalidEmbeddingVectorError,
        match="向量维度必须一致",
    ):
        FakeEmbeddingClient(
            {
                "短向量": (1.0, 0.0),
                "长向量": (1.0, 0.0, 0.0),
            }
        )
