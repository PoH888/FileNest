"""多格式文档加载与规范化。"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4
from zipfile import BadZipFile

from .document_contracts import Document, DocumentPage, DocumentPosition
from .filesystem_adapter import FileSystemAdapter


class DocumentParseError(ValueError):
    """文档无法按对应格式解析。"""


class UnsupportedDocumentFormatError(DocumentParseError):
    """没有 loader 能处理来源文件格式。"""


class DocumentLoader(Protocol):
    """所有文档格式共用的加载协议。"""

    def supports(self, source_relative_path: str | Path) -> bool:
        """判断 loader 是否负责给定来源路径。"""

    def load(
        self,
        adapter: FileSystemAdapter,
        *,
        source_relative_path: str | Path,
    ) -> "LoadedDocumentContent":
        """从已授权文件提取统一的文档内容。"""


DocumentSourceFormat = Literal["markdown", "text", "pdf", "docx"]


@dataclass(frozen=True, slots=True)
class LoadedDocumentContent:
    """loader 提取的格式内容，供统一入口组装为 Document。"""

    source_format: DocumentSourceFormat
    normalized_text: str
    pages: tuple[DocumentPage, ...] = ()
    source_positions: tuple[DocumentPosition, ...] = ()


def normalize_document_text(text: str) -> str:
    """只统一 BOM 与换行符，不改写 Markdown/TXT 的正文空白。"""

    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


class TextDocumentLoader:
    """加载 Markdown/TXT，并将格式差异限制在 loader 内部。"""

    _SOURCE_FORMATS: dict[str, Literal["markdown", "text"]] = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
    }

    def supports(self, source_relative_path: str | Path) -> bool:
        """按扩展名判断是否由文本 loader 处理。"""

        return Path(source_relative_path).suffix.casefold() in self._SOURCE_FORMATS

    def load(
        self,
        adapter: FileSystemAdapter,
        *,
        source_relative_path: str | Path,
    ) -> LoadedDocumentContent:
        """通过已授权适配器提取 Markdown/TXT 正文。"""

        source_path = Path(source_relative_path)
        source_format = _source_format_for_path(source_path)
        try:
            raw_text = adapter.read_text(source_path, encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("document must be valid UTF-8 text") from exc

        return LoadedDocumentContent(
            source_format=source_format,
            normalized_text=normalize_document_text(raw_text),
        )


class PdfDocumentLoader:
    """提取 PDF 页面文本，并保留每页在规范化正文中的位置。"""

    def supports(self, source_relative_path: str | Path) -> bool:
        """按扩展名判断是否由 PDF loader 处理。"""

        return Path(source_relative_path).suffix.casefold() == ".pdf"

    def load(
        self,
        adapter: FileSystemAdapter,
        *,
        source_relative_path: str | Path,
    ) -> LoadedDocumentContent:
        """通过已授权路径提取 PDF 各页文本。"""

        source_path = Path(source_relative_path)
        if not self.supports(source_path):
            suffix = source_path.suffix.casefold()
            raise UnsupportedDocumentFormatError(
                f"unsupported document format: {suffix or '<none>'}"
            )

        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:
            raise DocumentParseError("PDF loader requires the pypdf package") from exc

        authorized_path = adapter.authorized_path(source_path)
        try:
            reader = PdfReader(str(authorized_path))
            page_texts = [
                normalize_document_text(page.extract_text() or "")
                for page in reader.pages
            ]
        except (OSError, PdfReadError, ValueError, KeyError, IndexError) as exc:
            raise DocumentParseError("document is not a readable PDF") from exc

        if not page_texts:
            raise DocumentParseError("PDF document must contain at least one page")

        normalized_text_parts: list[str] = []
        pages: list[DocumentPage] = []
        current_offset = 0
        for page_number, page_text in enumerate(page_texts, start=1):
            start_offset = current_offset
            normalized_text_parts.append(page_text)
            current_offset += len(page_text)
            pages.append(
                DocumentPage(
                    page_number=page_number,
                    start_offset=start_offset,
                    end_offset=current_offset,
                )
            )
            if page_number < len(page_texts):
                normalized_text_parts.append("\n\n")
                current_offset += 2

        return LoadedDocumentContent(
            source_format="pdf",
            normalized_text="".join(normalized_text_parts),
            pages=tuple(pages),
        )


@dataclass(frozen=True, slots=True)
class _DocxTextBlock:
    """DOCX 正文块及其 OOXML 结构位置。"""

    text: str
    element_type: Literal["paragraph", "table_cell"]
    section_index: int | None = None
    heading_level: int | None = None
    paragraph_index: int | None = None
    table_index: int | None = None
    row_index: int | None = None
    cell_index: int | None = None


_HEADING_STYLE_PATTERN = re.compile(
    r"^heading\s*([1-9][0-9]*)$",
    re.IGNORECASE,
)


def _heading_level_for_paragraph(paragraph: object) -> int | None:
    """从 DOCX 段落样式提取 heading 层级。"""

    style = getattr(paragraph, "style", None)
    if style is None:
        return None

    for identifier in (
        getattr(style, "style_id", None),
        getattr(style, "name", None),
    ):
        if not isinstance(identifier, str):
            continue
        match = _HEADING_STYLE_PATTERN.fullmatch(identifier.strip())
        if match is not None:
            return int(match.group(1))
    return None


def _paragraph_ends_section(paragraph: object) -> bool:
    """判断段落是否携带结束当前 DOCX section 的 sectPr。"""

    paragraph_element = getattr(paragraph, "_p", None)
    paragraph_properties = getattr(paragraph_element, "pPr", None)
    return getattr(paragraph_properties, "sectPr", None) is not None


class DocxDocumentLoader:
    """提取 DOCX 段落和表格单元格，并保留结构位置。"""

    def supports(self, source_relative_path: str | Path) -> bool:
        """按扩展名判断是否由 DOCX loader 处理。"""

        return Path(source_relative_path).suffix.casefold() == ".docx"

    def load(
        self,
        adapter: FileSystemAdapter,
        *,
        source_relative_path: str | Path,
    ) -> LoadedDocumentContent:
        """通过已授权路径提取 DOCX 正文块。"""

        source_path = Path(source_relative_path)
        if not self.supports(source_path):
            suffix = source_path.suffix.casefold()
            raise UnsupportedDocumentFormatError(
                f"unsupported document format: {suffix or '<none>'}"
            )

        try:
            from docx import Document as WordDocument
            from docx.opc.exceptions import PackageNotFoundError
            from docx.oxml.exceptions import InvalidXmlError
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DocumentParseError(
                "DOCX loader requires the python-docx package"
            ) from exc

        authorized_path = adapter.authorized_path(source_path)
        try:
            word_document = WordDocument(str(authorized_path))
            blocks: list[_DocxTextBlock] = []
            paragraph_index = 0
            table_index = 0
            section_index = 0

            for element in word_document.element.body.iterchildren():
                element_type = element.tag.rsplit("}", 1)[-1]
                if element_type == "p":
                    paragraph = Paragraph(element, word_document)
                    blocks.append(
                        _DocxTextBlock(
                            text=paragraph.text,
                            element_type="paragraph",
                            section_index=section_index,
                            heading_level=_heading_level_for_paragraph(paragraph),
                            paragraph_index=paragraph_index,
                        )
                    )
                    paragraph_index += 1
                    if _paragraph_ends_section(paragraph):
                        section_index += 1
                elif element_type == "tbl":
                    table = Table(element, word_document)
                    for row_index, row in enumerate(table.rows):
                        for cell_index, cell in enumerate(row.cells):
                            blocks.append(
                                _DocxTextBlock(
                                    text=cell.text,
                                    element_type="table_cell",
                                    section_index=section_index,
                                    table_index=table_index,
                                    row_index=row_index,
                                    cell_index=cell_index,
                                )
                            )
                    table_index += 1
        except (
            OSError,
            BadZipFile,
            PackageNotFoundError,
            InvalidXmlError,
            ValueError,
            KeyError,
            IndexError,
        ) as exc:
            raise DocumentParseError("document is not a readable DOCX") from exc

        normalized_text_parts: list[str] = []
        source_positions: list[DocumentPosition] = []
        current_offset = 0
        for block_index, block in enumerate(blocks):
            if block_index > 0:
                normalized_text_parts.append("\n\n")
                current_offset += 2

            block_text = normalize_document_text(block.text)
            start_offset = current_offset
            normalized_text_parts.append(block_text)
            current_offset += len(block_text)
            source_positions.append(
                DocumentPosition(
                    element_type=block.element_type,
                    start_offset=start_offset,
                    end_offset=current_offset,
                    section_index=block.section_index,
                    heading_level=block.heading_level,
                    paragraph_index=block.paragraph_index,
                    table_index=block.table_index,
                    row_index=block.row_index,
                    cell_index=block.cell_index,
                )
            )

        return LoadedDocumentContent(
            source_format="docx",
            normalized_text="".join(normalized_text_parts),
            source_positions=tuple(source_positions),
        )


_DOCUMENT_LOADERS: tuple[DocumentLoader, ...] = (
    TextDocumentLoader(),
    PdfDocumentLoader(),
    DocxDocumentLoader(),
)


def load_document(
    adapter: FileSystemAdapter,
    *,
    workspace_id: int,
    file_entry_id: int,
    source_relative_path: str | Path,
    document_id: UUID | None = None,
) -> Document:
    """通过统一 loader 协议加载一个受支持格式的 Document。"""

    source_path = Path(source_relative_path)
    for loader in _DOCUMENT_LOADERS:
        if loader.supports(source_path):
            source_metadata = adapter.get_file_metadata(source_path)
            if source_metadata is None:
                raise DocumentParseError("source path must point to a regular file")

            loaded_content = loader.load(
                adapter,
                source_relative_path=source_path,
            )
            source_version = adapter.get_file_sha256(source_path)
            if source_version is None:
                raise DocumentParseError("source path must remain a regular file")

            return Document(
                document_id=document_id if document_id is not None else uuid4(),
                workspace_id=workspace_id,
                file_entry_id=file_entry_id,
                source_relative_path=source_path.as_posix(),
                source_format=loaded_content.source_format,
                normalized_text=loaded_content.normalized_text,
                pages=loaded_content.pages,
                source_positions=loaded_content.source_positions,
                source_version=source_version,
                source_updated_at=datetime.fromtimestamp(
                    source_metadata.mtime_ns / 1_000_000_000,
                    tz=timezone.utc,
                ),
            )

    suffix = source_path.suffix.casefold()
    raise UnsupportedDocumentFormatError(
        f"unsupported document format: {suffix or '<none>'}"
    )


def parse_document(
    adapter: FileSystemAdapter,
    *,
    workspace_id: int,
    file_entry_id: int,
    source_relative_path: str | Path,
    document_id: UUID | None = None,
) -> Document:
    """兼容旧调用方的统一文档加载入口。"""

    return load_document(
        adapter,
        workspace_id=workspace_id,
        file_entry_id=file_entry_id,
        source_relative_path=source_relative_path,
        document_id=document_id,
    )


def source_format_for_path(source_relative_path: str | Path) -> DocumentSourceFormat:
    """按统一 ingestion 入口支持的扩展名返回格式名。"""

    suffix = Path(source_relative_path).suffix.casefold()
    text_format = TextDocumentLoader._SOURCE_FORMATS.get(suffix)
    if text_format is not None:
        return text_format
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    raise UnsupportedDocumentFormatError(
        f"unsupported document format: {suffix or '<none>'}"
    )


def _source_format_for_path(path: Path) -> Literal["markdown", "text"]:
    """将文本 loader 支持的扩展名映射为统一格式名。"""

    suffix = path.suffix.casefold()
    source_format = TextDocumentLoader._SOURCE_FORMATS.get(suffix)
    if source_format is not None:
        return source_format
    raise UnsupportedDocumentFormatError(
        f"unsupported document format: {suffix or '<none>'}"
    )
