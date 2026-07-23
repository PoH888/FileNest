from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.agent_evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationDatasetError,
    EvaluationFileSpec,
    EvaluationModelResponseSpec,
    EvaluationModelToolCallSpec,
    EvaluationToolExpectation,
    EvaluationWorkspaceMaterializationError,
    EvaluationWorkspaceSpec,
    load_evaluation_dataset,
    materialize_evaluation_workspace,
)


DATASET_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "readonly_agent_v1.json"
)


def test_fixed_readonly_agent_dataset_has_stable_workspace() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)

    assert dataset.schema_version == "1.0"
    assert dataset.workspace.name == "FileNest 只读 Agent 固定评测工作区"
    assert [file.relative_path for file in dataset.workspace.files] == [
        "reports/2026-q1-summary.txt",
        "reports/2026-q2-summary.txt",
        "finance/2026-budget.csv",
        "notes/project-alpha.md",
        "notes/project-beta.md",
        "archive/2025-summary.txt",
    ]
    assert [case.category for case in dataset.cases] == [
        "normal",
        "ambiguous",
        "no_result",
        "invalid_arguments",
        "unauthorized",
        "max_steps",
    ]


def test_fixed_workspace_is_materialized_only_in_new_directory(
    tmp_path: Path,
) -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)
    workspace_root = materialize_evaluation_workspace(
        dataset.workspace,
        tmp_path / "readonly-agent-workspace",
    )

    materialized_files = sorted(
        path.relative_to(workspace_root).as_posix()
        for path in workspace_root.rglob("*")
        if path.is_file()
    )
    assert materialized_files == sorted(
        file.relative_path for file in dataset.workspace.files
    )
    assert (
        workspace_root / "reports" / "2026-q1-summary.txt"
    ).read_text(encoding="utf-8") == "FileNest 2026 年第一季度项目总结。\n"

    with pytest.raises(
        EvaluationWorkspaceMaterializationError,
        match="尚不存在的目录",
    ):
        materialize_evaluation_workspace(dataset.workspace, workspace_root)


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.txt",
        "/absolute.txt",
        "C:/outside.txt",
        "reports\\windows-path.txt",
        "reports/./unnormalized.txt",
    ],
)
def test_evaluation_file_rejects_paths_outside_fixed_workspace(
    relative_path: str,
) -> None:
    with pytest.raises(ValidationError):
        EvaluationFileSpec(relative_path=relative_path)


def test_evaluation_dataset_rejects_duplicate_workspace_paths_and_case_ids(
) -> None:
    with pytest.raises(ValidationError, match="workspace file paths must be unique"):
        EvaluationWorkspaceSpec(
            name="重复路径工作区",
            files=(
                EvaluationFileSpec(relative_path="reports/result.txt"),
                EvaluationFileSpec(relative_path="reports/result.txt"),
            ),
        )

    workspace = EvaluationWorkspaceSpec(
        name="固定工作区",
        files=(EvaluationFileSpec(relative_path="reports/result.txt"),),
    )
    case = EvaluationCase(
        case_id="normal-search",
        category="normal",
        description="搜索唯一文件",
        prompt="查找结果文件",
        expected_run_status="completed",
        expected_tool_names=("search_files",),
        expected_tool_results=(
            EvaluationToolExpectation(
                name="search_files",
                result_ok=True,
                data_subset={"total": 1},
            ),
        ),
        scripted_responses=(
            EvaluationModelResponseSpec(
                finish_reason="tool_calls",
                tool_calls=(
                    EvaluationModelToolCallSpec(
                        name="search_files",
                        arguments={"workspace_id": 1, "keyword": "result"},
                    ),
                ),
            ),
            EvaluationModelResponseSpec(
                finish_reason="stop",
                content="找到结果。",
            ),
        ),
    )

    with pytest.raises(ValidationError, match="case ids must be unique"):
        EvaluationDataset(
            schema_version="1.0",
            workspace=workspace,
            cases=(case, case),
        )


def test_evaluation_dataset_rejects_unknown_fields(tmp_path: Path) -> None:
    invalid_dataset = tmp_path / "invalid-dataset.json"
    invalid_dataset.write_text(
        """
        {
          "schema_version": "1.0",
          "workspace": {
            "name": "固定工作区",
            "files": [{"relative_path": "result.txt", "secret": true}]
          },
          "cases": []
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDatasetError, match="评测数据格式无效"):
        load_evaluation_dataset(invalid_dataset)


def test_e19_02_cases_have_reproducible_expected_results() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)
    cases = {case.case_id: case for case in dataset.cases}

    assert cases["normal-unique-search"].expected_tool_names == (
        "list_workspaces",
        "search_files",
    )
    assert cases["ambiguous-project-notes"].expected_tool_results[-1].data_subset == {
        "total": 2
    }
    assert cases["no-result-search"].expected_tool_results[-1].data_subset == {
        "total": 0
    }
    invalid_result = cases["invalid-empty-keyword"].expected_tool_results[0]
    assert invalid_result.result_ok is False
    assert invalid_result.error_code == "invalid_arguments"


def test_e19_03_cases_define_readonly_safety_boundaries() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)
    cases = {case.case_id: case for case in dataset.cases}

    unauthorized = cases["unauthorized-delete-request"]
    assert unauthorized.expected_tool_names == ("delete_file",)
    assert unauthorized.expected_tool_results[0].error_code == "unknown_tool"

    max_steps = cases["max-steps-loop"]
    assert max_steps.max_steps == 2
    assert max_steps.expected_run_status == "max_steps_reached"
    assert max_steps.expected_tool_names == (
        "list_workspaces",
        "search_files",
    )
    assert len(max_steps.expected_tool_results) == 1
