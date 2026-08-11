from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query # Depends：告诉 FastAPI：执行这个路由前，先调用指定的依赖函数，并把结果交给路由。
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from sqlalchemy.orm import Session

from .agent_api import router as agent_router
from .organization_api import router as organization_router
from .models import FileEntry, Workspace
from .database import check_database_connection, get_session
# get_session()：负责为每次 HTTP 请求创建和关闭 Session。

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

app = FastAPI(title="FileNest API")
app.include_router(agent_router)
app.include_router(organization_router)
_MINIMAL_UI_PATH = Path(__file__).parent / "static" / "index.html"


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
    response_model=FileIndexSyncResponse,
)
def scan_workspace(
    workspace_id: int,
    session: Session = Depends(get_session),
) -> FileIndexSyncResult:
    """安全扫描工作区，并将完整结果同步到文件索引。"""

    try:
        return scan_workspace_service(session, workspace_id)
    except WorkspaceNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        ) from error
    except WorkspaceScanUnavailableError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "workspace_scan_unavailable",
                "message": "工作区目录当前不可扫描。",
            },
        ) from error
