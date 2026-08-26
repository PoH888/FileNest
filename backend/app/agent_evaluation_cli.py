"""生成只读 Agent 里程碑评测结果与 Markdown 报告。"""

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import subprocess

from .agent_evaluation import EvaluationVersionInfo, load_evaluation_dataset
from .agent_evaluation_runner import (
    EvaluationHistoryError,
    EvaluationSummary,
    append_evaluation_history,
    run_evaluation_dataset,
)
from .model_settings import ModelSettings
from .openai_compatible_model_client import OpenAICompatibleModelClient


DEFAULT_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "readonly_agent_v1.json"
)
DEFAULT_PROMPT_VERSION = "readonly_agent_prompt_v1"
REPRODUCTION_COMMAND = (
    r".\.venv\Scripts\python.exe -m backend.app.agent_evaluation_cli "
    r"--output-dir backend\backups\e19-readonly-agent-v1 "
    r"--report-path backend\evaluation\readonly_agent_milestone.md"
)


class EvaluationMetadataError(RuntimeError):
    """评测版本信息无法安全取得。"""


def _current_git_commit(repo_root: Path) -> str:
    """只读取当前仓库提交标识，不读取或修改工作区状态。"""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        raise EvaluationMetadataError(
            "无法读取评测所需的 Git commit"
        ) from None

    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        raise EvaluationMetadataError(
            "无法读取评测所需的 Git commit"
        )
    return commit.casefold()


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
    parser.add_argument(
        "--model-source",
        choices=("scripted_fake", "real_model"),
        default="scripted_fake",
        help="选择确定性 Fake Model 或环境变量配置的真实模型。",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="本次评测使用的 Prompt 版本标识。",
    )
    parser.add_argument(
        "--evaluation-dataset-version",
        default=None,
        help="评测数据集版本标识，默认使用数据集文件名。",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help="跨运行评测历史 JSONL 路径，默认与报告同目录。",
    )
    return parser


def render_milestone_report(
    summary: EvaluationSummary,
    *,
    dataset_sha256: str,
) -> str:
    """把不含提示词和工具载荷的安全结果渲染为报告。"""

    metrics = summary.metrics
    reproduction_command = REPRODUCTION_COMMAND
    boundary_note = "- 本报告评测确定性的程序边界，不代表真实模型质量。"
    cost_note = "- Fake Model 不产生外部费用。"
    if summary.model_source == "real_model":
        reproduction_command += " --model-source real_model"
        boundary_note = "- 本报告包含真实模型调用结果，结果受模型版本和服务状态影响。"
        cost_note = "- 真实模型费用仅在供应商用量和价格信息完整时估算。"
    lines = [
        "# FileNest 第 19 课：只读 Agent 评测里程碑",
        "",
        "## 评测边界",
        "",
        f"- 数据格式版本：`{summary.dataset_schema_version}`",
        f"- 数据集 SHA-256：`{dataset_sha256}`",
        f"- 模型来源：`{summary.model_source}`",
        boundary_note,
        "- 报告不保存提示词、工具参数、工具返回载荷或绝对路径。",
        "",
        "## Evaluation Version",
        "",
        f"- Prompt version：`{summary.version_info.prompt_version}`",
        f"- Model version：`{summary.version_info.model_version}`",
        f"- Git commit：`{summary.version_info.git_commit}`",
        (
            "- Evaluation dataset version："
            f"`{summary.version_info.evaluation_dataset_version}`"
        ),
        f"- Timestamp：`{summary.version_info.timestamp.isoformat()}`",
        "",
        "## Forbidden Tools",
        "",
        (
            "- 检查结果："
            f"{'通过' if summary.forbidden_tools.passed else '失败'}"
        ),
        (
            "- 禁止工具："
            f"{', '.join(summary.forbidden_tools.forbidden_tool_names) or '无'}"
        ),
        (
            "- 未授权工具："
            f"{', '.join(summary.forbidden_tools.unapproved_tool_names) or '无'}"
        ),
        "",
        "## Risk Constraint Evaluation",
        "",
        (
            "- 检查结果："
            f"{'通过' if summary.risk_constraints.passed else '失败'}"
        ),
        (
            "- 检查用例："
            f"{', '.join(summary.risk_constraints.checked_case_ids) or '无'}"
        ),
        (
            "- 失败用例："
            f"{', '.join(summary.risk_constraints.failed_case_ids) or '无'}"
        ),
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
            "- 延迟是本机离线实测值，会随环境变化。",
            cost_note,
            "",
            "## 可复现命令",
            "",
            "在仓库根目录、且目标路径尚不存在时运行：",
            "",
            "```powershell",
            reproduction_command,
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
    history_path = args.history_path or args.report_path.with_name(
        "evaluation-history.jsonl"
    )
    if not history_path.parent.is_dir():
        raise FileNotFoundError("评测历史文件的父目录不存在")
    if history_path.is_symlink():
        raise EvaluationHistoryError("拒绝向符号链接追加评测历史")
    if history_path.resolve() == args.report_path.resolve():
        raise EvaluationHistoryError("评测历史路径不能与报告路径相同")

    dataset_bytes = args.dataset.read_bytes()
    dataset = load_evaluation_dataset(args.dataset)
    model_client_factory = None
    model_version = "scripted_fake"
    if args.model_source == "real_model":
        settings = ModelSettings()
        real_model_client = OpenAICompatibleModelClient(settings)
        model_version = settings.name

        def create_model_client() -> OpenAICompatibleModelClient:
            return real_model_client

        model_client_factory = create_model_client

    version_info = EvaluationVersionInfo(
        prompt_version=args.prompt_version,
        model_version=model_version,
        git_commit=_current_git_commit(Path(__file__).resolve().parents[2]),
        evaluation_dataset_version=(
            args.evaluation_dataset_version or args.dataset.stem
        ),
        timestamp=datetime.now(timezone.utc),
    )

    summary = run_evaluation_dataset(
        dataset,
        args.output_dir,
        model_client_factory=model_client_factory,
        model_source=args.model_source,
        version_info=version_info,
    )
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
    append_evaluation_history(summary, history_path)

    print(f"评测结果：{args.output_dir / 'evaluation-result.json'}")
    print(f"里程碑报告：{args.report_path}")
    print(f"评测历史：{history_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
