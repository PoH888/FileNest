"""工作区数据访问层。

集中封装原本写在 API 路由中的数据库查询与持久化操作，
使路由只负责接收请求、调用业务能力和生成 HTTP 响应。
"""

from sqlalchemy.orm import Session
from sqlalchemy import select # select()：构造数据库查询。

from .models import Workspace

def get_workspace_by_id(
        session: Session, # 从外部传入
        workspace_id: int # 要查询的主键
) -> Workspace | None:
    """按 ID 查询一个工作区，找不到时返回 None。"""

    return session.get(Workspace, workspace_id) # 按主键查询


def find_workspaces(
    session: Session,
    name: str | None = None,
) -> list[Workspace]:
    """查询工作区列表；传入名称时只返回同名工作区。"""

    statement = select(Workspace)

    if name is not None:
        statement = statement.where(Workspace.name == name)

    return list(session.scalars(statement).all())

def add_workspace(
    session: Session,
    workspace: Workspace,
) -> None:
    """将工作区加入当前 Session，等待后续提交。"""

    session.add(workspace) # 把对象登记为“等待保存”
