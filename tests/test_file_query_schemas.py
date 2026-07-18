from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    FileDetailResponse,
    FileListItemResponse,
    FileListResponse,
    FileQueryParams,
    FileSortField,
    SortOrder,
)


def test_file_query_defaults_and_normalizes_text_filters() -> None:
    query = FileQueryParams(
        keyword="  Quarterly Report  ",
        extension=" PDF ",
    )

    assert query.keyword == "Quarterly Report"
    assert query.extension == ".pdf"
    assert query.sort_by is FileSortField.RELATIVE_PATH
    assert query.sort_order is SortOrder.ASC
    assert query.page == 1
    assert query.page_size == 50


def test_file_query_accepts_sort_pagination_and_time_range() -> None:
    query = FileQueryParams(
        modified_from="2026-08-01T00:00:00+08:00",
        modified_to="2026-08-31T23:59:59+08:00",
        sort_by="modified_at",
        sort_order="desc",
        page=2,
        page_size=25,
    )

    assert query.modified_from is not None
    assert query.modified_to is not None
    assert query.modified_from < query.modified_to
    assert query.sort_by is FileSortField.MODIFIED_AT
    assert query.sort_order is SortOrder.DESC
    assert query.page == 2
    assert query.page_size == 25


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("keyword", "   "),
        ("extension", "   "),
        ("extension", "x" * 50),
        ("sort_by", "unknown"),
        ("sort_order", "newest"),
        ("page", 0),
        ("page_size", 0),
        ("page_size", 101),
        ("modified_from", "2026-08-01T00:00:00"),
    ],
)
def test_file_query_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        FileQueryParams(**{field: value})


def test_file_query_rejects_reversed_time_range() -> None:
    with pytest.raises(ValidationError):
        FileQueryParams(
            modified_from="2026-09-01T00:00:00+08:00",
            modified_to="2026-08-01T00:00:00+08:00",
        )


def test_file_response_rejects_time_without_timezone() -> None:
    with pytest.raises(ValidationError):
        FileListItemResponse(
            id=7,
            relative_path="report.pdf",
            name="report.pdf",
            extension=".pdf",
            size_bytes=2048,
            modified_at=datetime(2026, 8, 30, 9, 30),
        )


def test_file_list_and_detail_responses_expose_only_index_metadata() -> None:
    modified_at = datetime(2026, 8, 30, 9, 30, tzinfo=timezone.utc)
    item = FileListItemResponse(
        id=7,
        relative_path="documents/report.pdf",
        name="report.pdf",
        extension=".pdf",
        size_bytes=2048,
        modified_at=modified_at,
    )
    response = FileListResponse(
        items=[item],
        total=1,
        page=1,
        page_size=50,
    )
    detail = FileDetailResponse(
        **item.model_dump(),
        workspace_id=3,
    )

    assert response.model_dump() == {
        "items": [
            {
                "id": 7,
                "relative_path": "documents/report.pdf",
                "name": "report.pdf",
                "extension": ".pdf",
                "size_bytes": 2048,
                "modified_at": modified_at,
            }
        ],
        "total": 1,
        "page": 1,
        "page_size": 50,
    }
    assert detail.workspace_id == 3
    assert "root_path" not in detail.model_dump()
