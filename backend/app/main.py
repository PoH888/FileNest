from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Query # Depends：告诉 FastAPI：执行这个路由前，先调用指定的依赖函数，并把结果交给路由。
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .agent_api import router as agent_router
from .agent_recovery import recover_unfinished_agent_runs
from .document_indexer import (
    DocumentIndexWorkspaceNotFoundError,
    index_workspace_documents,
)
from .job_runner import JobContext, JobNotFoundError, JobTaskError, SingleProcessJobRunner
from .job_store import SqlAlchemyJobStore
from .knowledge_api import router as knowledge_router
from .organization_api import router as organization_router
from .models import FileEntry, Workspace
from .database import SessionFactory, check_database_connection, get_session
from .path_policy import PathPolicyError
# get_session()：负责为每次 HTTP 请求创建和关闭 Session。
from .safe_execution import (
    recover_unfinished_operation_executions,
    scan_unfinished_operation_executions,
)

from .services import (
    FileEntryNotFoundError,
    FileIndexSyncResult,
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
    application.state.unfinished_operation_execution_snapshots = ()
    application.state.recovered_operation_execution_results = ()
    application.state.job_runtimes = {}
    application.state.job_runtime_lock = RLock()
    try:
        with SessionFactory() as session:
            application.state.unfinished_agent_run_ids = (
                recover_unfinished_agent_runs(session)
            )
    except SQLAlchemyError:
        # 旧数据库尚未完成迁移时，保持应用可启动，由健康检查报告存储问题。
        application.state.unfinished_agent_run_ids = ()
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


class JobStatusResponse(BaseModel):
    """Job 状态查询的公开投影。"""

    job_id: UUID
    status: str
    error_code: str | None = None


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
            runtime = _JobRuntime(
                runner=SingleProcessJobRunner(
                    store=SqlAlchemyJobStore(task_session_factory),
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


def _public_job_status(status: str) -> str:
    """将内部 succeeded 状态转换为 API 约定的 completed。"""

    return "completed" if status == "succeeded" else status


def _submit_document_index_job(
    session: Session,
    workspace_id: int,
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
    submitted = runtime.runner.submit(
        kind="document_index",
        workspace_id=workspace_id,
        idempotency_key=f"document-index:{workspace_id}:{uuid4()}",
        task=lambda context: _run_document_index(
            context,
            session_factory=runtime.session_factory,
            workspace_id=workspace_id,
        ),
    )
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
    submitted = runtime.runner.submit(
        kind="workspace_scan",
        workspace_id=workspace_id,
        idempotency_key=f"workspace-scan:{workspace_id}:{uuid4()}",
        task=lambda context: _run_workspace_scan(
            context,
            session_factory=runtime.session_factory,
            workspace_id=workspace_id,
        ),
    )
    return JobSubmissionResponse(job_id=submitted.job_id)


@app.post(
    "/api/v1/workspaces/{workspace_id}/documents/index",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def index_documents(
    workspace_id: int,
    session: Session = Depends(get_session),
) -> JobSubmissionResponse:
    """创建后台文档索引 Job，并立即返回其标识。"""

    return _submit_document_index_job(session, workspace_id)


@app.post(
    "/api/v1/knowledge/index",
    response_model=JobSubmissionResponse,
    status_code=202,
)
def index_knowledge(
    request: KnowledgeIndexRequest,
    session: Session = Depends(get_session),
) -> JobSubmissionResponse:
    """创建 Knowledge 文档索引 Job，并立即返回其标识。"""

    return _submit_document_index_job(session, request.workspace_id)


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: UUID,
    session: Session = Depends(get_session),
) -> JobStatusResponse:
    """返回指定 Job 的当前状态。"""

    runtime = _job_runtime_for_session(session)
    try:
        state = runtime.runner.get(job_id)
    except JobNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "job_not_found",
                "message": "Job 不存在。",
            },
        ) from error

    return JobStatusResponse(
        job_id=state.job_id,
        status=_public_job_status(state.status),
        error_code=state.error_code,
    )
