from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Annotated, cast
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
) # Depends：告诉 FastAPI：执行这个路由前，先调用指定的依赖函数，并把结果交给路由。
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .approval_recovery import (
    ApprovalRecoveryIssue,
    ApprovalRecoveryErrorCode,
    scan_waiting_approval_tasks,
)
from .agent_api import router as agent_router
from .agent_recovery import (
    recover_unfinished_agent_runs,
    scan_agent_recovery_snapshots,
)
from .document_indexer import (
    DocumentIndexWorkspaceNotFoundError,
    index_workspace_documents,
)
from .evaluation_api import router as evaluation_router
from .job_runner import (
    JobContext,
    JobActionConflictError,
    JobHandlerRegistry,
    JobIdentityConflictError,
    JobNotFoundError,
    JobTaskError,
    SingleProcessJobRunner,
)
from .job_store import JobStoreError, SqlAlchemyJobStore
from .job_system import (
    JobKind,
    JobState,
    JobStatus,
    JobTaskPayload,
)
from .knowledge_api import router as knowledge_router
from .organization_api import router as organization_router
from .models import FileEntry, Workspace
from .database import SessionFactory, check_database_connection, get_session
from .path_policy import PathPolicyError, validate_workspace_root
# get_session()：负责为每次 HTTP 请求创建和关闭 Session。
from .safe_execution import (
    recover_unfinished_operation_executions,
    scan_unfinished_operation_executions,
)
from .workflow_graph import open_checkpointed_workflow_graph
from .workflow_runtime import WORKFLOW_CHECKPOINT_PATH

from .services import (
    FileEntryNotFoundError,
    FileSearchResult,
    WorkspacePathConflictError,
    WorkspaceNotFoundError,
    WorkspaceScanUnavailableError,
    create_workspace as create_workspace_service,
    get_file_detail as get_file_detail_service,
    get_workspace as get_workspace_service,
    list_workspaces as list_workspaces_service,
    scan_workspace as scan_workspace_service,
    search_files as search_files_service,
)
from .schemas import (
    FileDetailResponse,
    FileListItemResponse,
    FileListResponse,
    FileQueryParams,
)



@asynccontextmanager
async def _lifespan(application: FastAPI):
    """服务启动时记录未完成 AgentRun，并恢复未收尾 Execution。"""

    application.state.unfinished_agent_run_ids = ()
    application.state.agent_recovery_snapshots = ()
    application.state.unfinished_operation_execution_snapshots = ()
    application.state.recovered_operation_execution_results = ()
    application.state.recovered_job_ids = ()
    application.state.approval_recovery_snapshots = ()
    application.state.approval_recovery_issues = ()
    application.state.approval_recovery_error_count = 0
    application.state.job_runtimes = {}
    application.state.job_runtime_lock = RLock()
    try:
        with SessionFactory() as session:
            application.state.unfinished_agent_run_ids = (
                recover_unfinished_agent_runs(session)
            )
            application.state.agent_recovery_snapshots = (
                scan_agent_recovery_snapshots(session)
            )
    except SQLAlchemyError:
        # 旧数据库尚未完成迁移时，保持应用可启动，由健康检查报告存储问题。
        application.state.unfinished_agent_run_ids = ()
        application.state.agent_recovery_snapshots = ()
    try:
        with SessionFactory() as session:
            with open_checkpointed_workflow_graph(
                WORKFLOW_CHECKPOINT_PATH,
            ) as graph:
                approval_scan = scan_waiting_approval_tasks(session, graph)
        application.state.approval_recovery_snapshots = (
            approval_scan.recovered_tasks
        )
        application.state.approval_recovery_issues = approval_scan.issues
        application.state.approval_recovery_error_count = len(
            approval_scan.issues
        )
    except (OSError, SQLAlchemyError, ValueError):
        application.state.approval_recovery_snapshots = ()
        application.state.approval_recovery_issues = (
            ApprovalRecoveryIssue(
                approval_id=None,
                workflow_id=None,
                plan_id=None,
                workspace_id=None,
                code=ApprovalRecoveryErrorCode.RECOVERY_UNAVAILABLE.value,
            ),
        )
        application.state.approval_recovery_error_count = 1
    try:
        with SessionFactory() as session:
            runtime = _job_runtime_for_session(session)
            recovered_jobs = runtime.runner.recover_persisted_jobs(
                can_run=lambda state: _can_recover_job(
                    state,
                    runtime.session_factory,
                )
            )
            application.state.recovered_job_ids = tuple(
                state.job_id for state in recovered_jobs
            )
    except (JobStoreError, SQLAlchemyError):
        # 任务描述或数据库不可安全读取时，不猜测执行内容。
        application.state.recovered_job_ids = ()
    try:
        with SessionFactory() as session:
            snapshots = scan_unfinished_operation_executions(session)
            application.state.unfinished_operation_execution_snapshots = snapshots
            application.state.recovered_operation_execution_results = (
                recover_unfinished_operation_executions(session)
            )
    except SQLAlchemyError:
        # 旧数据库尚未完成执行状态迁移时，不阻断服务启动。
        application.state.unfinished_operation_execution_snapshots = ()
        application.state.recovered_operation_execution_results = ()
    try:
        yield
    finally:
        with application.state.job_runtime_lock:
            runtimes = tuple(application.state.job_runtimes.values())
            application.state.job_runtimes.clear()
        for runtime in runtimes:
            runtime.runner.shutdown()


app = FastAPI(title="FileNest API", lifespan=_lifespan)
app.include_router(agent_router)
app.include_router(evaluation_router)
app.include_router(knowledge_router)
app.include_router(organization_router)
_MINIMAL_UI_PATH = Path(__file__).parent / "static" / "index.html"


@dataclass(frozen=True, slots=True)
class _JobRuntime:
    runner: SingleProcessJobRunner
    session_factory: Callable[[], Session]


class JobSubmissionResponse(BaseModel):
    """后台 Job 提交成功后的最小公开标识。"""

    job_id: UUID


class KnowledgeIndexRequest(BaseModel):
    """Knowledge 索引请求。"""

    workspace_id: int


class JobAttemptResponse(BaseModel):
    """Job 详情中不含堆栈的 Attempt 投影。"""

    attempt_id: UUID
    attempt_no: int
    status: str
    completed_units: int
    total_units: int | None
    phase_code: str
    started_at: datetime
    finished_at: datetime | None
    error_code: str | None
    retryable: bool


class JobStatusResponse(BaseModel):
    """Job 查询的公开投影，不返回异常堆栈或 Python 对象。"""

    job_id: UUID
    kind: str
    workspace_id: int
    task_version: str
    payload_hash: str
    status: str
    error_code: str | None = None
    attempts: list[JobAttemptResponse] = Field(default_factory=list)


def _job_runtime_for_session(session: Session) -> _JobRuntime:
    """为当前数据库绑定复用单进程 Runner，后台任务使用独立 Session。"""

    bind = session.get_bind()
    with app.state.job_runtime_lock:
        runtime = app.state.job_runtimes.get(bind)
        if runtime is None:
            task_session_factory = sessionmaker(
                bind=bind,
                expire_on_commit=False,
            )
            handler_registry = _job_handler_registry(task_session_factory)
            runtime = _JobRuntime(
                runner=SingleProcessJobRunner(
                    store=SqlAlchemyJobStore(task_session_factory),
                    handler_registry=handler_registry,
                ),
                session_factory=task_session_factory,
            )
            app.state.job_runtimes[bind] = runtime
        return runtime


def _run_workspace_scan(
    _context: JobContext,
    *,
    session_factory: Callable[[], Session],
    workspace_id: int,
) -> None:
    """在后台 Session 中执行扫描；业务失败转换为稳定 Job 错误码。"""

    with session_factory() as task_session:
        try:
            scan_workspace_service(task_session, workspace_id)
        except WorkspaceNotFoundError as error:
            raise JobTaskError("workspace_not_found") from error
        except WorkspaceScanUnavailableError as error:
            raise JobTaskError("workspace_scan_unavailable") from error


def _run_document_index(
    _context: JobContext,
    *,
    session_factory: Callable[[], Session],
    workspace_id: int,
) -> None:
    """在后台 Session 中完成文档解析、分块和持久化。"""

    with session_factory() as task_session:
        try:
            index_workspace_documents(task_session, workspace_id)
        except DocumentIndexWorkspaceNotFoundError as error:
            raise JobTaskError("workspace_not_found") from error
        except Exception as error:
            raise JobTaskError("document_index_failed") from error


def _job_handler_registry(
    session_factory: Callable[[], Session],
) -> JobHandlerRegistry:
    """只注册当前代码明确支持的可重建任务版本。"""

    def run_workspace_scan(
        context: JobContext,
        payload: JobTaskPayload,
    ) -> None:
        _run_workspace_scan(
            context,
            session_factory=session_factory,
            workspace_id=payload.workspace_id,
        )

    def run_document_index(
        context: JobContext,
        payload: JobTaskPayload,
    ) -> None:
        _run_document_index(
            context,
            session_factory=session_factory,
            workspace_id=payload.workspace_id,
        )

    return JobHandlerRegistry(
        {
            ("workspace_scan", "v1"): run_workspace_scan,
            ("document_index", "v1"): run_document_index,
        }
    )


def _can_recover_job(
    state: JobState,
    session_factory: Callable[[], Session],
) -> bool:
    """恢复前重新确认工作区存在且当前根路径仍通过 PathPolicy。"""

    with session_factory() as session:
        workspace = get_workspace_service(session, state.workspace_id)
        if workspace is None:
            return False
        try:
            validate_workspace_root(workspace.root_path)
        except PathPolicyError:
            return False
    return True


def _public_job_status(status: str) -> str:
    """将内部 succeeded 状态转换为 API 约定的 completed。"""

    return "completed" if status == "succeeded" else status


def _job_status_response(state: JobState) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=state.job_id,
        kind=state.kind,
        workspace_id=state.workspace_id,
        task_version=state.task_version,
        payload_hash=state.payload_hash or "",
        status=_public_job_status(state.status),
        error_code=state.error_code,
        attempts=[
            JobAttemptResponse(
                attempt_id=attempt.attempt_id,
                attempt_no=attempt.attempt_no,
                status=attempt.status,
                completed_units=attempt.progress.completed_units,
                total_units=attempt.progress.total_units,
                phase_code=attempt.progress.phase_code,
                started_at=attempt.started_at,
                finished_at=attempt.finished_at,
                error_code=attempt.error_code,
                retryable=attempt.retryable,
            )
            for attempt in state.attempts
        ],
    )


def _job_not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "code": "job_not_found",
            "message": "Job 不存在。",
        },
    )


def _job_action_conflict(error: JobActionConflictError) -> HTTPException:
    messages = {
        "job_cancel_not_allowed": "当前 Job 不允许取消。",
        "job_retry_not_allowed": "当前 Job 不允许重试。",
        "job_state_changed": "Job 状态已改变，请重新读取后再操作。",
    }
    return HTTPException(
        status_code=409,
        detail={
            "code": error.code,
            "message": messages.get(error.code, "Job 操作冲突。"),
        },
    )


def _normalize_job_kind(value: str | None) -> JobKind | None:
    if value is None:
        return None
    if value not in {"workspace_scan", "document_index"}:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_job_kind",
                "message": "Job kind 不受支持。",
            },
        )
    return cast(JobKind, value)


def _normalize_job_status(value: str | None) -> JobStatus | None:
    if value is None:
        return None
    if value == "completed":
        return "succeeded"
    if value not in {
        "pending",
        "running",
        "cancel_requested",
        "succeeded",
        "failed",
        "cancelled",
    }:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_job_status",
                "message": "Job status 不受支持。",
            },
        )
    return cast(JobStatus, value)


def _resolve_idempotency_key(
    *,
    kind: str,
    workspace_id: int,
    client_key: str | None,
) -> str:
    """将网络重试身份与随机 Job ID 分离。"""

    if client_key is None:
        return f"{kind}:v1:workspace:{workspace_id}"
    if (
        not client_key
        or len(client_key) > 128
        or client_key != client_key.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_idempotency_key",
                "message": "Idempotency-Key 必须是 1 至 128 个无首尾空格的字符。",
            },
        )
    return client_key


def _job_identity_conflict(error: JobIdentityConflictError) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "code": "job_identity_conflict",
            "message": "Idempotency-Key 已绑定到不同的 Job 定义。",
        },
    )


def _submit_document_index_job(
    session: Session,
    workspace_id: int,
    client_idempotency_key: str | None,
) -> JobSubmissionResponse:
    """校验工作区并提交可复用的文档索引 Job。"""

    if get_workspace_service(session, workspace_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    runtime = _job_runtime_for_session(session)
    try:
        submitted = runtime.runner.submit(
            kind="document_index",
            workspace_id=workspace_id,
            idempotency_key=_resolve_idempotency_key(
                kind="document_index",
                workspace_id=workspace_id,
                client_key=client_idempotency_key,
            ),
            payload=JobTaskPayload(workspace_id=workspace_id),
        )
    except JobIdentityConflictError as error:
        raise _job_identity_conflict(error) from error
    return JobSubmissionResponse(job_id=submitted.job_id)


@app.get("/", include_in_schema=False)
def minimal_ui() -> FileResponse:
    """返回不绕过 HTTP API 的最小单页界面。"""

    return FileResponse(_MINIMAL_UI_PATH, media_type="text/html")

class WorkspaceCreate(BaseModel):
    """校验客户端提交的数据"""
    name: str
    root_path: str

class WorkspaceResponse(BaseModel):
    """校验服务端返回的数据"""

    # Workspace ORM 对象→读取 .id、.name、.root_path→WorkspaceResponse→JSON
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    root_path: str


class FileIndexSyncResponse(BaseModel):
    """一次工作区扫描产生的索引变化统计。"""

    model_config = ConfigDict(from_attributes=True)

    created: int
    updated: int
    deleted: int
    unchanged: int



# 健康检查
@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    """返回服务当前可用的健康状态。"""

    if not check_database_connection():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "database_unavailable",
                "message": "数据库不可用。",
            },
        )

    return {"status": "ok"}

# 添加工作区
@app.post("/api/v1/workspaces",
          status_code=201,
          response_model=WorkspaceResponse,
          ) # 返回数据必须符合 WorkspaceResponse 的结构，并由 FastAPI进行校验、序列化和文档展示。
def create_workspace(
        workspace: WorkspaceCreate,
        session: Session = Depends(get_session),
) -> Workspace:
    """接收创建工作区请求，并将业务处理交给 Service。"""

    try:
        return create_workspace_service( # 即services.py中的create_workspace()
            session,
            workspace.name,
            workspace.root_path,
        )
    except WorkspacePathConflictError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workspace_path_conflict",
                "message": "工作区路径已存在。",
            },
        ) from error
    except PathPolicyError as error:
        raise HTTPException(
            status_code=409,
            detail=error.as_detail(),
        ) from error

# 查询工作区列表
@app.get("/api/v1/workspaces", response_model=list[WorkspaceResponse])
                                                 # 对外响应契约
def list_workspaces(
        name: str | None = None,
        # str | None：类型声明：允许出现 None, = None：默认值：没传参数时使用 None
        session: Session = Depends(get_session),
        #                  不从客户端请求中读取 session，而是调用 get_session()获得
) -> list[Workspace]: # 数据库内部对象
    """返回工作区列表，可按名称筛选。"""

    return list_workspaces_service(session, name)

# 支持按名称筛选
@app.get("/api/v1/workspaces/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
        workspace_id: int,
        session: Session = Depends(get_session),
) -> Workspace:
    """返回指定工作区；不存在时返回 404。"""

    workspace = get_workspace_service(session, workspace_id)

    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    return workspace


@app.get(
    "/api/v1/workspaces/{workspace_id}/files",
    response_model=FileListResponse,
)
def list_files(
    workspace_id: int,
    query: Annotated[FileQueryParams, Query()],
    session: Session = Depends(get_session),
) -> FileListResponse:
    """返回工作区文件索引，可搜索、过滤、排序和分页。"""

    try:
        result = search_files_service(
            session,
            workspace_id,
            keyword=query.keyword,
            extension=query.extension,
            modified_from=query.modified_from,
            modified_to=query.modified_to,
            sort_by=query.sort_by.value,
            sort_order=query.sort_order.value,
            page=query.page,
            page_size=query.page_size,
        )
    except WorkspaceNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        ) from error

    return _file_list_response(result)


@app.get(
    "/api/v1/workspaces/{workspace_id}/files/{file_id}",
    response_model=FileDetailResponse,
)
def get_file_detail(
    workspace_id: int,
    file_id: int,
    session: Session = Depends(get_session),
) -> FileDetailResponse:
    """返回指定工作区内一个文件索引的详情。"""

    try:
        file_entry = get_file_detail_service(session, workspace_id, file_id)
    except WorkspaceNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        ) from error
    except FileEntryNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "file_not_found",
                "message": "文件索引不存在。",
            },
        ) from error

    item = _file_list_item_response(file_entry)
    return FileDetailResponse(
        **item.model_dump(),
        workspace_id=file_entry.workspace_id,
    )


def _file_list_response(result: FileSearchResult) -> FileListResponse:
    """将内部文件索引转换为不含绝对路径的公开响应。"""

    return FileListResponse(
        items=[_file_list_item_response(item) for item in result.items],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
    )


def _file_list_item_response(file_entry: FileEntry) -> FileListItemResponse:
    seconds, nanoseconds = divmod(file_entry.mtime_ns, 1_000_000_000)
    modified_at = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanoseconds // 1_000,
    )
    return FileListItemResponse(
        id=file_entry.id,
        relative_path=file_entry.relative_path,
        name=file_entry.name,
        extension=file_entry.extension,
        size_bytes=file_entry.size_bytes,
        modified_at=modified_at,
    )


@app.post(
    "/api/v1/workspaces/{workspace_id}/scan",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def scan_workspace(
    workspace_id: int,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
    session: Session = Depends(get_session),

) -> JobSubmissionResponse:
    """创建后台扫描 Job，并立即返回其标识。"""

    if get_workspace_service(session, workspace_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    runtime = _job_runtime_for_session(session)
    try:
        submitted = runtime.runner.submit(
            kind="workspace_scan",
            workspace_id=workspace_id,
            idempotency_key=_resolve_idempotency_key(
                kind="workspace_scan",
                workspace_id=workspace_id,
                client_key=idempotency_key,
            ),
            payload=JobTaskPayload(workspace_id=workspace_id),
        )
    except JobIdentityConflictError as error:
        raise _job_identity_conflict(error) from error
    return JobSubmissionResponse(job_id=submitted.job_id)


@app.post(
    "/api/v1/workspaces/{workspace_id}/documents/index",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def index_documents(
    workspace_id: int,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
    session: Session = Depends(get_session),
) -> JobSubmissionResponse:
    """创建后台文档索引 Job，并立即返回其标识。"""

    return _submit_document_index_job(
        session,
        workspace_id,
        idempotency_key,
    )


@app.post(
    "/api/v1/knowledge/index",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def index_knowledge(
    request: KnowledgeIndexRequest,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
    session: Session = Depends(get_session),
) -> JobSubmissionResponse:
    """创建 Knowledge 文档索引 Job，并立即返回其标识。"""

    return _submit_document_index_job(
        session,
        request.workspace_id,
        idempotency_key,
    )


@app.get("/api/v1/jobs", response_model=list[JobStatusResponse])
def list_jobs(
    workspace_id: int = Query(..., ge=1),
    kind: str | None = Query(default=None),
    status: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[JobStatusResponse]:
    """按工作区列出 Job，不允许省略租户边界。"""

    runtime = _job_runtime_for_session(session)
    states = runtime.runner.list_jobs(
        workspace_id=workspace_id,
        kind=_normalize_job_kind(kind),
        status=_normalize_job_status(status),
    )
    return [_job_status_response(state) for state in states]


def _get_job_for_workspace(
    runtime: _JobRuntime,
    job_id: UUID,
    workspace_id: int,
) -> JobState:
    try:
        state = runtime.runner.get(job_id)
    except JobNotFoundError as error:
        raise _job_not_found() from error
    if state.workspace_id != workspace_id:
        raise _job_not_found()
    return state


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: UUID,
    workspace_id: int = Query(..., ge=1),
    session: Session = Depends(get_session),
) -> JobStatusResponse:
    """返回指定工作区内 Job 的当前状态和 Attempt 历史。"""

    runtime = _job_runtime_for_session(session)
    state = _get_job_for_workspace(runtime, job_id, workspace_id)
    return _job_status_response(state)


@app.post(
    "/api/v1/jobs/{job_id}/cancel",
    response_model=JobStatusResponse,
)
def cancel_job(
    job_id: UUID,
    workspace_id: int = Query(..., ge=1),
    session: Session = Depends(get_session),
) -> JobStatusResponse:
    """请求协作式取消，不强杀后台线程。"""

    runtime = _job_runtime_for_session(session)
    _get_job_for_workspace(runtime, job_id, workspace_id)
    try:
        state = runtime.runner.cancel(job_id)
    except JobNotFoundError as error:
        raise _job_not_found() from error
    except JobActionConflictError as error:
        raise _job_action_conflict(error) from error
    return _job_status_response(state)


@app.post(
    "/api/v1/jobs/{job_id}/retry",
    response_model=JobStatusResponse,
)
def retry_job(
    job_id: UUID,
    workspace_id: int = Query(..., ge=1),
    session: Session = Depends(get_session),
) -> JobStatusResponse:
    """请求对可重试失败 Job 创建新的 Attempt。"""

    runtime = _job_runtime_for_session(session)
    _get_job_for_workspace(runtime, job_id, workspace_id)
    try:
        state = runtime.runner.retry(job_id)
    except JobNotFoundError as error:
        raise _job_not_found() from error
    except JobActionConflictError as error:
        raise _job_action_conflict(error) from error
    return _job_status_response(state)
