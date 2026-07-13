# FastAPI → Service → Repository → Session → SQLite

"""工作区业务服务层。

负责组织完整的工作区业务流程和事务，
不处理 HTTP 路由、状态码或响应格式。
"""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import Workspace
from .repositories import add_workspace


class WorkspacePathConflictError(Exception):
    """工作区根路径已经存在。"""


def create_workspace(
    session: Session,
    name: str,
    root_path: str,
) -> Workspace:
    workspace = Workspace(
        name=name,
        root_path=root_path,
    )

    add_workspace(session, workspace)

    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise WorkspacePathConflictError from error

    return workspace