"""按文件身份与来源内容版本识别文档变化。"""

from typing import Literal, TypeAlias

from .document_contracts import Document


DocumentVersionKey: TypeAlias = tuple[int, str]
DocumentVersionChange: TypeAlias = Literal["new", "modified", "duplicate"]


def document_version_key(document: Document) -> DocumentVersionKey:
    """返回可用于增量导入比较的文件版本键。"""

    return document.file_entry_id, document.source_version


def classify_document_version(
    current: Document,
    previous: Document | None,
) -> DocumentVersionChange:
    """根据同一文件的来源版本识别新增、修改或重复导入。"""

    if previous is None or current.file_entry_id != previous.file_entry_id:
        return "new"
    if current.source_version == previous.source_version:
        return "duplicate"
    return "modified"
