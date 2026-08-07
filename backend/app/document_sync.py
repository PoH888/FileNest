"""对文档快照做一次性增量差异计算。"""

from collections.abc import Sequence
from dataclasses import dataclass

from .document_contracts import Document
from .document_versioning import classify_document_version


class DuplicateDocumentSnapshotError(ValueError):
    """同一快照中一个文件出现多个文档版本。"""


@dataclass(frozen=True, slots=True)
class DocumentSyncResult:
    """一次文档快照同步的分类结果。"""

    new_documents: tuple[Document, ...]
    modified_documents: tuple[Document, ...]
    deleted_documents: tuple[Document, ...]
    duplicate_documents: tuple[Document, ...]


def sync_document_snapshot(
    current_documents: Sequence[Document],
    previous_documents: Sequence[Document],
) -> DocumentSyncResult:
    """比较两次每文件一个版本的快照，不执行文件或数据库操作。"""

    current_by_file = _index_snapshot(current_documents)
    previous_by_file = _index_snapshot(previous_documents)

    new_documents: list[Document] = []
    modified_documents: list[Document] = []
    duplicate_documents: list[Document] = []

    for document in current_documents:
        previous = previous_by_file.get(document.file_entry_id)
        change = classify_document_version(document, previous)
        if change == "new":
            new_documents.append(document)
        elif change == "modified":
            modified_documents.append(document)
        else:
            duplicate_documents.append(document)

    deleted_documents = [
        document
        for document in previous_documents
        if document.file_entry_id not in current_by_file
    ]

    return DocumentSyncResult(
        new_documents=tuple(new_documents),
        modified_documents=tuple(modified_documents),
        deleted_documents=tuple(deleted_documents),
        duplicate_documents=tuple(duplicate_documents),
    )


def _index_snapshot(documents: Sequence[Document]) -> dict[int, Document]:
    """建立文件身份索引，避免重复版本让同步结果产生歧义。"""

    indexed: dict[int, Document] = {}
    for document in documents:
        if document.file_entry_id in indexed:
            raise DuplicateDocumentSnapshotError(
                "snapshot contains multiple documents for one file entry"
            )
        indexed[document.file_entry_id] = document
    return indexed
