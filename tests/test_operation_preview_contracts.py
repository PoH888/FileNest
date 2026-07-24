import pytest
from pydantic import ValidationError

from backend.app.operation_preview import (
    OperationPreviewCandidate,
    OperationPreviewItem,
    OperationPreviewRequest,
    OperationPreviewResponse,
)


def test_operation_preview_contracts_describe_read_only_ranked_candidates() -> None:
    request = OperationPreviewRequest(
        workspace_id=3,
        source_file_ids=[7, 8],
        target_directories=["documents", "documents/reports"],
    )
    response = OperationPreviewResponse(
        workspace_id=request.workspace_id,
        items=[
            OperationPreviewItem(
                source_file_id=7,
                source_relative_path="inbox/quarterly-report.pdf",
                candidates=[
                    OperationPreviewCandidate(
                        relative_directory="documents/reports",
                        score=96,
                    ),
                    OperationPreviewCandidate(
                        relative_directory="documents",
                        score=72,
                    ),
                ],
            ),
            OperationPreviewItem(
                source_file_id=8,
                source_relative_path="inbox/unknown.bin",
            ),
        ],
    )

    assert request.source_file_ids == (7, 8)
    assert response.read_only is True
    assert response.items[0].candidates[0].score == 96
    assert response.items[1].candidates == ()
    assert "selected_destination" not in response.model_dump_json()
    assert "root_path" not in response.model_dump_json()


@pytest.mark.parametrize(
    "target_directories",
    [
        ["documents", "documents"],
        ["../outside"],
        ["D:/outside"],
        ["documents\\reports"],
        [" documents"],
        ["."],
    ],
)
def test_preview_request_rejects_duplicate_or_unsafe_target_directories(
    target_directories: list[str],
) -> None:
    with pytest.raises(ValidationError):
        OperationPreviewRequest(
            workspace_id=1,
            source_file_ids=[1],
            target_directories=target_directories,
        )


def test_preview_request_rejects_duplicate_source_ids_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        OperationPreviewRequest(
            workspace_id=1,
            source_file_ids=[4, 4],
            target_directories=["documents"],
        )

    with pytest.raises(ValidationError):
        OperationPreviewRequest(
            workspace_id=1,
            source_file_ids=[4],
            target_directories=["documents"],
            execute=True,
        )


def test_preview_item_requires_unique_candidates_in_descending_score_order() -> None:
    with pytest.raises(ValidationError, match="descending score"):
        OperationPreviewItem(
            source_file_id=1,
            source_relative_path="inbox/report.pdf",
            candidates=[
                {"relative_directory": "reports", "score": 70},
                {"relative_directory": "archive", "score": 90},
            ],
        )

    with pytest.raises(ValidationError, match="candidate directories must be unique"):
        OperationPreviewItem(
            source_file_id=1,
            source_relative_path="inbox/report.pdf",
            candidates=[
                {"relative_directory": "reports", "score": 90},
                {"relative_directory": "reports", "score": 70},
            ],
        )


def test_preview_response_rejects_execution_state_and_duplicate_files() -> None:
    item = OperationPreviewItem(
        source_file_id=1,
        source_relative_path="inbox/report.pdf",
    )

    with pytest.raises(ValidationError):
        OperationPreviewResponse(
            workspace_id=1,
            items=[item],
            read_only=False,
        )

    with pytest.raises(ValidationError, match="unique source_file_ids"):
        OperationPreviewResponse(
            workspace_id=1,
            items=[item, item],
        )
