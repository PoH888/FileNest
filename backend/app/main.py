from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="FileNest API")

class WorkspaceCreate(BaseModel):
    """约束客户端提交的数据"""
    name: str
    root_path: str

class WorkspaceResponse(BaseModel):
    """约束服务端返回的数据"""
    id: int
    name: str
    root_path: str

workspaces: list[WorkspaceResponse] = []

# 健康检查
@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

# 添加工作区
@app.post("/api/v1/workspaces",
          status_code=201,
          response_model=WorkspaceResponse,
          ) # 返回数据必须符合 WorkspaceResponse 的结构，并由 FastAPI进行校验、序列化和文档展示。
def create_workspace(workspace: WorkspaceCreate) -> WorkspaceResponse:
    """收到创建请求
    → 检查路径是否重复
    → 重复则返回 409
    → 不重复则创建 WorkspaceResponse
    → 保存到 workspaces
    → 返回创建结果"""

    for existing_workspace in workspaces:
        if existing_workspace.root_path == workspace.root_path:
            raise HTTPException( # HTTP 错误响应异常
                status_code=409,
                detail={
                    "code": "workspace_path_conflict",
                    "message": "工作区路径已存在。",
                }
            )

    # 继承BaseModel，可检查字段是否齐全、类型是否符合要求等
    created_workspace = WorkspaceResponse(
        id=len(workspaces) + 1,
        name=workspace.name,
        root_path=workspace.root_path,
    )
    workspaces.append(created_workspace)
    return created_workspace

# 查询工作区列表
@app.get("/api/v1/workspaces", response_model=list[WorkspaceResponse])
def list_workspaces(
        name: str | None = None,
        # str | None：类型声明：允许出现 None, = None：默认值：没传参数时使用 None
) -> list[WorkspaceResponse]:
    if name is None:
        return workspaces
    return [workspace for workspace in workspaces if workspace.name == name]

# 支持按名称筛选
@app.get("/api/v1/workspaces/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(workspace_id: int) -> WorkspaceResponse:
    for workspace in workspaces:
        if workspace.id == workspace_id:
            return workspace

    raise HTTPException(
        status_code=404,
        detail={
            "code": "workspace_not_found",
            "message": "工作区路径已存在。"
        },
    )