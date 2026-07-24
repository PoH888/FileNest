import pytest

from backend.app.operation_preview import rank_preview_candidates


def test_rank_preview_candidates_wraps_legacy_matcher_without_selecting() -> None:
    candidates = rank_preview_candidates(
        "Python_Project_Code.zip",
        (
            "programming/Python",
            "programming/Java",
            "programming/C++",
        ),
    )

    assert candidates
    assert candidates[0].relative_directory == "programming/Python"
    assert [candidate.score for candidate in candidates] == sorted(
        (candidate.score for candidate in candidates),
        reverse=True,
    )
    candidate_directories = {
        candidate.relative_directory for candidate in candidates
    }
    assert "programming/Java" not in candidate_directories
    assert "programming/C++" not in candidate_directories


def test_rank_preview_candidates_preserves_multiple_ranked_candidates() -> None:
    candidates = rank_preview_candidates(
        "Tokyo_Travel_Plan.xlsx",
        (
            "travel/Japan Travel",
            "travel/Tokyo Trip",
            "work/Finance",
        ),
    )

    assert candidates[0].relative_directory == "travel/Tokyo Trip"
    assert "travel/Japan Travel" in {
        candidate.relative_directory for candidate in candidates
    }
    assert "work/Finance" not in {
        candidate.relative_directory for candidate in candidates
    }


@pytest.mark.parametrize(
    "source_file_name",
    ["xyzabcxyz_random_file.exe", "a.txt"],
)
def test_rank_preview_candidates_returns_empty_when_matcher_has_no_candidate(
    source_file_name: str,
) -> None:
    candidates = rank_preview_candidates(
        source_file_name,
        ("documents/Reports", "documents/Archive"),
    )

    assert candidates == ()


@pytest.mark.parametrize(
    ("source_file_name", "target_directories"),
    [
        (" report.pdf", ("documents",)),
        ("inbox/report.pdf", ("documents",)),
        ("report.pdf", ()),
        ("report.pdf", ("documents", "documents")),
        ("report.pdf", ("../outside",)),
    ],
)
def test_rank_preview_candidates_rejects_invalid_adapter_inputs(
    source_file_name: str,
    target_directories: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        rank_preview_candidates(source_file_name, target_directories)
