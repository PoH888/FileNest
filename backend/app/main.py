from fastapi import Depends, FastAPI, HTTPException # Depends：告诉 FastAPI：执行这个路由前，先调用指定的依赖函数，并把结果交给路由。
from pydantic import BaseModel, ConfigDict

from sqlalchemy.orm import Session

from .models import Workspace
from .database import get_session
# get_session()：负责为每次 HTTP 请求创建和关闭 Session。

from .services import (
    WorkspacePathConflictError,
    create_workspace as create_workspace_service,
    get_workspace as get_workspace_service,
    list_workspaces as list_workspaces_service,
)

app = FastAPI(title="FileNest API")

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



# 健康检查
@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
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
    return list_workspaces_service(session, name)

# 支持按名称筛选
@app.get("/api/v1/workspaces/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
        workspace_id: int,
        session: Session = Depends(get_session),
) -> Workspace:
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
