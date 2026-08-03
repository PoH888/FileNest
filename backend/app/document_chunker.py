"""将规范化文档按可解释规则切分为可追踪片段。"""

from uuid import uuid5

from .document_contracts import Chunk, Document


DEFAULT_MAX_CHARS = 800


def chunk_document(
    document: Document,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[Chunk, ...]:
    """按行贪心装箱，超长单行再按字符上限切分。"""

    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars < 1
    ):
        raise ValueError("max_chars must be a positive integer")

    chunks: list[Chunk] = []
    current_start: int | None = None
    current_end = 0
    current_start_line = 0
    current_end_line = 0
    next_chunk_index = 0
    offset = 0

    for line_number, line in enumerate(
        document.normalized_text.splitlines(keepends=True),
        start=1,
    ):
        line_start = offset
        line_end = line_start + len(line)
        offset = line_end

        if len(line) > max_chars:
            if current_start is not None:
                next_chunk_index = _append_chunk(
                    chunks,
                    document,
                    next_chunk_index,
                    current_start,
                    current_end,
                    current_start_line,
                    current_end_line,
                )
                current_start = None

            for segment_start in range(line_start, line_end, max_chars):
                segment_end = min(segment_start + max_chars, line_end)
                next_chunk_index = _append_chunk(
                    chunks,
                    document,
                    next_chunk_index,
                    segment_start,
                    segment_end,
                    line_number,
                    line_number,
                )
            continue

        if current_start is None:
            current_start = line_start
            current_end = line_end
            current_start_line = line_number
            current_end_line = line_number
        elif line_end - current_start <= max_chars:
            current_end = line_end
            current_end_line = line_number
        else:
            next_chunk_index = _append_chunk(
                chunks,
                document,
                next_chunk_index,
                current_start,
                current_end,
                current_start_line,
                current_end_line,
            )
            current_start = line_start
            current_end = line_end
            current_start_line = line_number
            current_end_line = line_number

    if current_start is not None:
        _append_chunk(
            chunks,
            document,
            next_chunk_index,
            current_start,
            current_end,
            current_start_line,
            current_end_line,
        )

    return tuple(chunks)


def _append_chunk(
    chunks: list[Chunk],
    document: Document,
    chunk_index: int,
    start_offset: int,
    end_offset: int,
    start_line: int,
    end_line: int,
) -> int:
    """从文档区间创建一个片段，并返回下一个顺序号。"""

    chunks.append(
        Chunk(
            chunk_id=uuid5(
                document.document_id,
                f"{document.source_version}:{chunk_index}",
            ),
            document_id=document.document_id,
            file_entry_id=document.file_entry_id,
            source_relative_path=document.source_relative_path,
            chunk_index=chunk_index,
            text=document.normalized_text[start_offset:end_offset],
            start_offset=start_offset,
            end_offset=end_offset,
            start_line=start_line,
            end_line=end_line,
        )
    )
    return chunk_index + 1
