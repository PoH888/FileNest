"""运行安全整理固定评测并生成不可静默覆盖的证据。"""

import argparse
from collections.abc import Sequence
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import cast
from xml.etree import ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAFE_CHAIN_TEST_PATH = Path("tests/test_safe_organization_end_to_end.py")
UNAPPROVED_TEST_PATH = Path("tests/test_approval_disk_immutability.py")
SOURCE_PATHS = (SAFE_CHAIN_TEST_PATH, UNAPPROVED_TEST_PATH)
EVALUATION_TARGETS = (
    SAFE_CHAIN_TEST_PATH.as_posix(),
    (
        f"{UNAPPROVED_TEST_PATH.as_posix()}::"
        "test_unapproved_status_leaves_complete_disk_snapshot_unchanged"
    ),
    (
        f"{UNAPPROVED_TEST_PATH.as_posix()}::"
        "test_missing_approval_leaves_complete_disk_snapshot_unchanged"
    ),
    (
        f"{UNAPPROVED_TEST_PATH.as_posix()}::"
        "test_mismatched_approved_plan_leaves_disk_unchanged"
    ),
)
SAFE_CHAIN_CASE_IDS = frozenset(
    {
        "test_query_plan_approve_execute_and_undo_real_file_chain",
        "test_approved_cross_workspace_plan_cannot_execute",
        "test_file_changed_after_approval_is_rejected_before_execution",
        "test_restart_replays_duplicate_without_move_and_can_undo",
        "test_batch_partial_failure_compensates_only_completed_move",
    }
)
UNAPPROVED_CASE_PREFIXES = (
    "test_unapproved_status_leaves_complete_disk_snapshot_unchanged",
    "test_missing_approval_leaves_complete_disk_snapshot_unchanged",
    "test_mismatched_approved_plan_leaves_disk_unchanged",
)
EXPECTED_SAFE_CHAIN_CASES = 5
EXPECTED_UNAPPROVED_CASES = 4
EXPECTED_CASES = EXPECTED_SAFE_CHAIN_CASES + EXPECTED_UNAPPROVED_CASES
REPRODUCTION_COMMAND = (
    r".\.venv\Scripts\python.exe "
    r"-m backend.app.safe_organization_evaluation_cli "
    r"--output-dir backend\backups\e26-safe-organization-v1 "
    r"--report-path backend\evaluation\safe_organization_milestone.md"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 FileNest 安全整理固定综合评测。"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="必须尚不存在的评测证据目录。",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="必须尚不存在的 Markdown 报告路径。",
    )
    return parser


def _source_hashes() -> dict[str, str]:
    return {
        path.as_posix(): sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _case_category(case_id: str) -> str:
    if case_id in SAFE_CHAIN_CASE_IDS:
        return "safe_chain"
    if case_id.startswith(UNAPPROVED_CASE_PREFIXES):
        return "unapproved_disk_immutability"
    raise RuntimeError(f"评测返回了未知用例：{case_id}")


def _read_case_results(junit_path: Path) -> list[dict[str, str]]:
    root = ElementTree.parse(junit_path).getroot()
    test_cases = root.findall(".//testcase")
    if len(test_cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"评测用例数量不匹配：期待 {EXPECTED_CASES}，实际 {len(test_cases)}"
        )

    results: list[dict[str, str]] = []
    for test_case in test_cases:
        case_id = test_case.attrib.get("name", "")
        if any(
            test_case.find(result_name) is not None
            for result_name in ("failure", "error", "skipped")
        ):
            raise RuntimeError(f"评测用例未通过：{case_id}")
        results.append(
            {
                "case_id": case_id,
                "category": _case_category(case_id),
                "result": "passed",
            }
        )

    safe_chain_count = sum(
        result["category"] == "safe_chain" for result in results
    )
    unapproved_count = sum(
        result["category"] == "unapproved_disk_immutability"
        for result in results
    )
    if safe_chain_count != EXPECTED_SAFE_CHAIN_CASES:
        raise RuntimeError("安全整理主链用例集合不完整")
    if unapproved_count != EXPECTED_UNAPPROVED_CASES:
        raise RuntimeError("未经审批磁盘快照用例集合不完整")
    actual_safe_chain_ids = {
        result["case_id"]
        for result in results
        if result["category"] == "safe_chain"
    }
    if actual_safe_chain_ids != SAFE_CHAIN_CASE_IDS:
        raise RuntimeError("安全整理主链用例名称与固定集合不一致")
    return results


def _run_fixed_evaluation(output_dir: Path) -> dict[str, object]:
    source_hashes_before = _source_hashes()
    output_dir.mkdir(parents=True)
    junit_path = output_dir / "pytest-results.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        *EVALUATION_TARGETS,
        "-q",
        f"--junitxml={junit_path}",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("安全整理固定评测未通过，拒绝生成成功报告")
    if not junit_path.is_file():
        raise RuntimeError("安全整理固定评测没有生成 JUnit 证据")

    cases = _read_case_results(junit_path)
    source_hashes_after = _source_hashes()
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("评测运行期间测试源文件发生变化，拒绝生成报告")

    return {
        "schema_version": "1.0",
        "suite": "safe_organization_v1",
        "case_total": EXPECTED_CASES,
        "passed_cases": EXPECTED_CASES,
        "safe_chain_cases": EXPECTED_SAFE_CHAIN_CASES,
        "unapproved_cases": EXPECTED_UNAPPROVED_CASES,
        "unauthorized_disk_changes": 0,
        "source_sha256": source_hashes_after,
        "cases": cases,
    }


def _save_summary(summary: dict[str, object], result_path: Path) -> None:
    with result_path.open("x", encoding="utf-8", newline="\n") as result_file:
        json.dump(summary, result_file, ensure_ascii=False, indent=2)
        result_file.write("\n")


def render_milestone_report(summary: dict[str, object]) -> str:
    source_hashes = cast(dict[str, str], summary["source_sha256"])
    cases = cast(list[dict[str, str]], summary["cases"])
    lines = [
        "# FileNest 第 26 课：安全整理 Agent 综合评测",
        "",
        "## 评测边界",
        "",
        "- 评测套件：`safe_organization_v1`",
        "- 使用临时 SQLite 与 pytest 临时工作区，不接触真实用户文件。",
        "- 本报告验证确定性的程序安全边界，不代表真实模型质量。",
        "- 报告不保存绝对路径、用户文件内容、提示词或工具载荷。",
        "",
        "## 评测源",
        "",
    ]
    for source_path, source_hash in source_hashes.items():
        lines.append(f"- `{source_path}` SHA-256：`{source_hash}`")

    lines.extend(
        [
            "",
            "## 汇总结果",
            "",
            "| 指标 | 结果 |",
            "| --- | ---: |",
            (
                f"| 综合安全场景通过 | {summary['passed_cases']}/"
                f"{summary['case_total']} |"
            ),
            (
                f"| 安全整理主链通过 | {summary['safe_chain_cases']}/"
                f"{EXPECTED_SAFE_CHAIN_CASES} |"
            ),
            (
                f"| 未经审批磁盘快照一致 | {summary['unapproved_cases']}/"
                f"{EXPECTED_UNAPPROVED_CASES} |"
            ),
            (
                "| 未经审批磁盘变更 | "
                f"{summary['unauthorized_disk_changes']} |"
            ),
            "",
            "## 用例结果",
            "",
            "| 用例 | 分类 | 结果 |",
            "| --- | --- | --- |",
        ]
    )
    for case in cases:
        lines.append(
            f"| `{case['case_id']}` | `{case['category']}` | 通过 |"
        )

    lines.extend(
        [
            "",
            "## 结论与限制",
            "",
            "- 4 个未经审批场景的完整磁盘快照前后相同，因此检测到的磁盘变更为 0。",
            "- 已验证查询、规划、审批、执行、undo、越界拒绝、文件变化拒绝、幂等、重启和部分失败补偿。",
            "- 重启场景使用全新 Engine/Session，不等同于完整服务进程重启。",
            "- 部分失败使用确定性的目标冲突注入，不代表生产并发压力测试。",
            "",
            "## 可复现命令",
            "",
            "在仓库根目录、且目标路径尚不存在时运行：",
            "",
            "```powershell",
            REPRODUCTION_COMMAND,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report_path.exists() or args.report_path.is_symlink():
        raise FileExistsError("安全评测报告已存在，拒绝覆盖历史证据")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("安全评测证据目录已存在，拒绝覆盖历史证据")
    if not args.report_path.parent.is_dir():
        raise FileNotFoundError("安全评测报告的父目录不存在")

    summary = _run_fixed_evaluation(args.output_dir)
    result_path = args.output_dir / "evaluation-result.json"
    _save_summary(summary, result_path)
    report = render_milestone_report(summary)
    with args.report_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as report_file:
        report_file.write(report)

    print(f"评测结果：{result_path}")
    print(f"里程碑报告：{args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
