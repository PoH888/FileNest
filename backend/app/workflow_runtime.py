"""FastAPI 使用的正式 workflow checkpoint 生命周期。"""

import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from .database import DATABASE_PATH, get_session
from .services import validate_operation_plan
from .workflow_graph import open_checkpointed_workflow_graph


_WORKFLOW_CHECKPOINT_PATH_ENV = "FILENEST_WORKFLOW_CHECKPOINT_PATH"


def _resolve_workflow_checkpoint_path() -> Path:
    """允许部署环境把 checkpoint 指向与业务数据库同一持久卷。"""

    configured_path = os.getenv(_WORKFLOW_CHECKPOINT_PATH_ENV)
    if configured_path and configured_path.strip():
        return Path(configured_path.strip())
    return DATABASE_PATH.with_name("workflow-checkpoints.sqlite")


WORKFLOW_CHECKPOINT_PATH = _resolve_workflow_checkpoint_path()


def get_workflow_graph(
    session: Session = Depends(get_session),
) -> Iterator[CompiledStateGraph]:
    """为一次请求打开独立于业务数据库的 checkpoint 连接。"""

    with open_checkpointed_workflow_graph(
        WORKFLOW_CHECKPOINT_PATH,
        operation_plan_validator=partial(validate_operation_plan, session),
    ) as graph:
        yield graph
