"""生成只读 Agent 里程碑评测结果与 Markdown 报告。"""

import argparse
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from .agent_evaluation import load_evaluation_dataset
from .agent_evaluation_runner import (
    EvaluationSummary,
    run_evaluation_dataset,
)


DEFAULT_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "readonly_agent_v1.json"
)
REPRODUCTION_COMMAND = (
    r".\.venv\Scripts\python.exe -m backend.app.agent_evaluation_cli "
    r"--output-dir backend\backups\e19-readonly-agent-v1 "
    r"--report-path backend\evaluation\readonly_agent_milestone.md"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 FileNest 只读 Agent 固定里程碑评测。"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="评测数据集 JSON 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="必须尚不存在的运行证据目录。",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="必须尚不存在的 Markdown 报告路径。",
    )
    return parser


def render_milestone_report(
    summary: EvaluationSummary,
    *,
    dataset_sha256: str,
) -> str:
    """把不含提示词和工具载荷的安全结果渲染为报告。"""

    metrics = summary.metrics
    lines = [
        "# FileNest 第 19 课：只读 Agent 评测里程碑",
        "",
        "## 评测边界",
        "",
        f"- 数据格式版本：`{summary.dataset_schema_version}`",
        f"- 数据集 SHA-256：`{dataset_sha256}`",
        f"- 模型来源：`{summary.model_source}`",
        "- 本报告评测确定性的程序边界，不代表真实模型质量。",
        "- 报告不保存提示词、工具参数、工具返回载荷或绝对路径。",
        "",
        "## 汇总结果",
        "",
        "| 指标 | 计数 | 比率 |",
        "| --- | ---: | ---: |",
        (
            f"| 任务成功率 | {metrics.successful_tasks}/"
            f"{metrics.task_total} | {metrics.task_success_rate:.2%} |"
        ),
        (
            f"| 工具选择率 | {metrics.correct_tool_selections}/"
            f"{metrics.tool_selection_total} | "
            f"{metrics.tool_selection_rate:.2%} |"
        ),
        (
            f"| 参数有效率 | {metrics.valid_parameter_calls}/"
            f"{metrics.parameter_call_total} | "
            f"{metrics.parameter_validity_rate:.2%} |"
        ),
        "",
        "## 用例结果",
        "",
        "| 用例 | 分类 | 运行状态 | 模型步数 | 结果 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for case in summary.cases:
        lines.append(
            f"| `{case.case_id}` | `{case.category}` | "
            f"`{case.actual_run_status}` | {case.model_turns} | "
            f"{'通过' if case.task_success else '失败'} |"
        )

    lines.extend(
        [
            "",
            "## 运行成本",
            "",
            f"- 总模型步数：{summary.total_model_turns}",
            f"- 总运行延迟：{summary.total_run_latency_ms:.3f} ms",
            (
                "- 预估模型费用："
                f"${summary.total_estimated_model_cost_usd} USD"
            ),
            "- 延迟是本机离线实测值，会随环境变化；Fake Model 不产生外部费用。",
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
        raise FileExistsError("里程碑报告已存在，拒绝覆盖历史证据")
    if not args.report_path.parent.is_dir():
        raise FileNotFoundError("里程碑报告的父目录不存在")

    dataset_bytes = args.dataset.read_bytes()
    dataset = load_evaluation_dataset(args.dataset)
    summary = run_evaluation_dataset(dataset, args.output_dir)
    report = render_milestone_report(
        summary,
        dataset_sha256=sha256(dataset_bytes).hexdigest(),
    )

    with args.report_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as report_file:
        report_file.write(report)

    print(f"评测结果：{args.output_dir / 'evaluation-result.json'}")
    print(f"里程碑报告：{args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
