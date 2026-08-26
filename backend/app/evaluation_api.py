"""固定只读 Agent Evaluation 的最小 HTTP 边界。"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path as ApiPath
from pydantic import BaseModel, ConfigDict, Field

from .agent_evaluation import (
    EvaluationVersionInfo,
    load_evaluation_dataset,
)
from .agent_evaluation_runner import (
    EvaluationCaseResult,
    EvaluationMetrics,
    EvaluationSummary,
    run_evaluation_dataset,
)


router = APIRouter(prefix="/api/v1")

DEFAULT_DATASET_PATH = (
    Path(__file__).parents[1]
    / "evaluation"
    / "readonly_agent_v1.json"
)
DEFAULT_PROMPT_VERSION = "readonly_agent_prompt_v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]


EvaluationApiRunStatus = Literal["pending", "running", "completed", "failed"]


class EvaluationRunAcceptedResponse(BaseModel):
    """创建 Evaluation run 后返回的公开句柄。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=1)


class EvaluationRunStatusResponse(BaseModel):
    """查询 Evaluation run 当前状态的公开投影。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=1)
    status: EvaluationApiRunStatus
    error_code: str | None = None


class EvaluationResultsResponse(BaseModel):
    """Evaluation run 结果的稳定公开投影。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: EvaluationSummary
    individual_cases: tuple[EvaluationCaseResult, ...]
    failures: tuple[EvaluationCaseResult, ...]
    metrics: EvaluationMetrics
    version_info: EvaluationVersionInfo


@dataclass
class _EvaluationRun:
    id: int
    status: EvaluationApiRunStatus = "pending"
    error_code: str | None = None
    summary: EvaluationSummary | None = None


_evaluation_runs: dict[int, _EvaluationRun] = {}
_evaluation_runs_lock = Lock()
_next_evaluation_id = 1
_evaluation_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="filenest-evaluation",
)


def _current_git_commit() -> str:
    """为结果版本信息读取当前提交，不公开 Git 命令的原始错误。"""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("无法读取评测所需的 Git commit") from error

    commit = result.stdout.strip()
    if len(commit) < 7 or len(commit) > 64:
        raise RuntimeError("无法读取评测所需的 Git commit")
    return commit.casefold()


def _build_version_info() -> EvaluationVersionInfo:
    return EvaluationVersionInfo(
        prompt_version=DEFAULT_PROMPT_VERSION,
        model_version="scripted_fake",
        git_commit=_current_git_commit(),
        evaluation_dataset_version=DEFAULT_DATASET_PATH.stem,
        timestamp=datetime.now(timezone.utc),
    )


def _create_evaluation_run() -> _EvaluationRun:
    global _next_evaluation_id

    with _evaluation_runs_lock:
        run = _EvaluationRun(id=_next_evaluation_id)
        _evaluation_runs[run.id] = run
        _next_evaluation_id += 1
    return run


def _evaluation_run_root(run_id: int) -> Path:
    """每次运行使用随机尚不存在的目录，避免覆盖历史评测证据。"""

    return Path(tempfile.gettempdir()) / f"filenest-evaluation-{run_id}-{uuid4().hex}"


def _set_run_failed(run_id: int, error_code: str) -> None:
    with _evaluation_runs_lock:
        run = _evaluation_runs.get(run_id)
        if run is not None:
            run.status = "failed"
            run.error_code = error_code


def _get_evaluation_run(run_id: int) -> _EvaluationRun | None:
    with _evaluation_runs_lock:
        return _evaluation_runs.get(run_id)


def _execute_evaluation(
    run_id: int,
    version_info: EvaluationVersionInfo,
    runner: Callable[..., EvaluationSummary] = run_evaluation_dataset,
) -> None:
    with _evaluation_runs_lock:
        run = _evaluation_runs.get(run_id)
        if run is None:
            return
        run.status = "running"

    try:
        dataset = load_evaluation_dataset(DEFAULT_DATASET_PATH)
        summary = runner(
            dataset,
            _evaluation_run_root(run_id),
            version_info=version_info,
        )
    except Exception:
        # 状态接口只需要稳定错误码，避免把本地路径或供应商错误泄露给调用方。
        _set_run_failed(run_id, "evaluation_failed")
        return

    with _evaluation_runs_lock:
        run = _evaluation_runs.get(run_id)
        if run is not None:
            run.status = "completed"
            run.summary = summary


def _submit_evaluation(
    run: _EvaluationRun,
    version_info: EvaluationVersionInfo,
) -> None:
    try:
        _evaluation_executor.submit(
            _execute_evaluation,
            run.id,
            version_info,
        )
    except RuntimeError as error:
        _set_run_failed(run.id, "evaluation_unavailable")
        raise error


@router.post(
    "/evaluations",
    status_code=202,
    response_model=EvaluationRunAcceptedResponse,
)
def create_evaluation_run() -> EvaluationRunAcceptedResponse:
    """创建固定数据集 Evaluation run，并立即返回运行句柄。"""

    try:
        version_info = _build_version_info()
        run = _create_evaluation_run()
        _submit_evaluation(run, version_info)
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "evaluation_unavailable",
                "message": "Evaluation 当前不可用。",
            },
        ) from error

    return EvaluationRunAcceptedResponse(id=run.id)


@router.get(
    "/evaluations/{run_id}",
    response_model=EvaluationRunStatusResponse,
)
def get_evaluation_run_status(
    run_id: int = ApiPath(ge=1),
) -> EvaluationRunStatusResponse:
    """读取 Evaluation run 状态，不介入运行。"""

    run = _get_evaluation_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "evaluation_not_found",
                "message": "Evaluation run 不存在。",
            },
        )

    return EvaluationRunStatusResponse(
        id=run.id,
        status=run.status,
        error_code=run.error_code,
    )


@router.get(
    "/evaluations/{run_id}/results",
    response_model=EvaluationResultsResponse,
)
def get_evaluation_results(
    run_id: int = ApiPath(ge=1),
) -> EvaluationResultsResponse:
    """读取已完成 Evaluation run 的汇总、用例和失败结果。"""

    run = _get_evaluation_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "evaluation_not_found",
                "message": "Evaluation run 不存在。",
            },
        )
    if run.status != "completed" or run.summary is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "evaluation_results_not_ready",
                "message": "Evaluation 结果尚未就绪。",
            },
        )

    summary = run.summary
    return EvaluationResultsResponse(
        summary=summary,
        individual_cases=summary.cases,
        failures=tuple(case for case in summary.cases if not case.task_success),
        metrics=summary.metrics,
        version_info=summary.version_info,
    )
