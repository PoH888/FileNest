from fastapi import Depends, FastAPI, HTTPException # Depends：告诉 FastAPI：执行这个路由前，先调用指定的依赖函数，并把结果交给路由。
from pydantic import BaseModel, ConfigDict

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import Workspace
from .database import Base, engine, get_session
# Base：保存所有已经登记的 ORM 表设计
# engine：告诉 SQLAlchemy要在哪个数据库中创建表
# get_session()：负责为每次 HTTP 请求创建和关闭 Session。

from .repositories import (add_workspace, find_workspaces, get_workspace_by_id,)

# 把 ORM 设计图和真实数据库连接起来
Base.metadata.create_all(bind=engine)
# Base.metadata：取得已经登记的表设计。
# create_all()：创建数据库中尚不存在的表。
# bind=engine：指定目标数据库是 engine 所连接的 SQLite。


app = FastAPI(title="FileNest API")

class WorkspaceCreate(BaseModel):
    """约束客户端提交的数据"""
    name: str
    root_path: str

class WorkspaceResponse(BaseModel):
    """约束服务端返回的数据"""

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
    """客户端提交 JSON
    → WorkspaceCreate 校验
    → 创建 Workspace ORM 对象
    → session.add()
    → session.commit()
    → SQLite 检查 root_path 唯一约束
    → 成功：生成 id，返回 201
    → 重复：rollback，返回 409
    → 请求结束，Session 关闭"""

    created_workspace = Workspace( # 创建 SQLAlchemy ORM 对象
        name=workspace.name, # 把请求对象中的 name 复制到 ORM 对象
        root_path=workspace.root_path, # 把请求对象中的 root_path 复制到 ORM 对象
    )

    add_workspace(session, created_workspace)

    try:
        session.commit() # 提交事务，成功后数据写入 filenest.db
    except IntegrityError as error:
        session.rollback() # 回滚失败的事务
        raise HTTPException( # # HTTP 错误响应异常
            status_code=409,
            detail={
                "code": "workspace_path_conflict",
                "message": "工作区路径已存在。",
            },
        ) from error # 抛出 HTTPException，同时保留原始 IntegrityError 作为异常原因

    return created_workspace

# 查询工作区列表
@app.get("/api/v1/workspaces", response_model=list[WorkspaceResponse])
                                                 # 对外响应契约
def list_workspaces(
        name: str | None = None,
        # str | None：类型声明：允许出现 None, = None：默认值：没传参数时使用 None
        session: Session = Depends(get_session),
        #                  不从客户端请求中读取 session，而是调用 get_session()获得
) -> list[Workspace]: # 数据库内部对象
    return find_workspaces(session, name)

# 支持按名称筛选
@app.get("/api/v1/workspaces/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(
        workspace_id: int,
        session: Session = Depends(get_session),
) -> Workspace:
    workspace = get_workspace_by_id(session, workspace_id)

    if workspace is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    return workspace