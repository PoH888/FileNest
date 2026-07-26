"""使用 LangGraph 编排并持久化 FileNest 的纯工作流状态转换。"""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .operation_plan import OperationPlan
from .workflow import (
    WorkflowEvent,
    WorkflowState,
    WorkflowTransitionError,
    WorkflowTransitionErrorCode,
    transition_workflow,
)


JsonObject = dict[str, object]
OperationPlanValidator = Callable[[OperationPlan], None]


class WorkflowBoundaryErrorCode(StrEnum):
    """工作流缺少必要业务边界时的稳定程序码。"""

    OPERATION_PLAN_VALIDATOR_REQUIRED = "operation_plan_validator_required"


class WorkflowBoundaryError(RuntimeError):
    """以失败关闭方式阻止 LangGraph 绕过业务 Service。"""

    def __init__(
        self,
        code: WorkflowBoundaryErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class WorkflowCheckpointErrorCode(StrEnum):
    """checkpoint 生命周期错误的稳定程序码。"""

    NOT_FOUND = "workflow_checkpoint_not_found"
    ALREADY_EXISTS = "workflow_checkpoint_already_exists"


class WorkflowCheckpointError(RuntimeError):
    """拒绝从缺失状态恢复或覆盖已有工作流。"""

    def __init__(
        self,
        code: WorkflowCheckpointErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class WorkflowGraphState(TypedDict):
    """只向 checkpoint 交付 JSON 安全数据，加载后再做业务校验。"""

    workflow: JsonObject
    event: JsonObject


def build_workflow_graph(
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    operation_plan_validator: OperationPlanValidator | None = None,
) -> CompiledStateGraph:
    """建立单节点图，并显式注入 checkpoint 与业务验证边界。"""

    builder = StateGraph(WorkflowGraphState)
    builder.add_node(
        "apply_event",
        lambda state: _apply_event(
            state,
            operation_plan_validator=operation_plan_validator,
        ),
    )
    builder.add_edge(START, "apply_event")
    builder.add_edge("apply_event", END)
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def open_checkpointed_workflow_graph(
    checkpoint_path: Path,
) -> Iterator[CompiledStateGraph]:
    """使用独立 SQLite 文件打开图，并明确管理数据库连接生命周期。"""

    if not checkpoint_path.parent.is_dir():
        raise ValueError("checkpoint parent directory does not exist")
    if checkpoint_path.exists() and not checkpoint_path.is_file():
        raise ValueError("checkpoint path must be a file")

    connection = sqlite3.connect(
        str(checkpoint_path),
        check_same_thread=False,
    )
    serializer = JsonPlusSerializer(
        allowed_json_modules=(),
        allowed_msgpack_modules=(),
    )
    checkpointer = SqliteSaver(connection, serde=serializer)
    try:
        yield build_workflow_graph(checkpointer=checkpointer)
    finally:
        connection.close()


def run_workflow_event(
    graph: CompiledStateGraph,
    workflow: WorkflowState,
    event: WorkflowEvent,
) -> WorkflowState:
    """运行一个事件，并在图输出离开边界前重新验证工作流状态。"""

    result = graph.invoke(
        WorkflowGraphState(
            workflow=workflow.model_dump(mode="json"),
            event=event.model_dump(mode="json"),
        )
    )
    return WorkflowState.model_validate(result["workflow"])


def run_checkpointed_workflow_event(
    graph: CompiledStateGraph,
    event: WorkflowEvent,
    *,
    workflow: WorkflowState | None = None,
) -> WorkflowState:
    """创建或恢复一个由 workflow_id 隔离的持久化工作流。"""

    config = workflow_checkpoint_config(event.workflow_id)
    saved_state = graph.get_state(config).values

    if workflow is None:
        if "workflow" not in saved_state:
            raise WorkflowCheckpointError(
                WorkflowCheckpointErrorCode.NOT_FOUND,
                "工作流 checkpoint 不存在",
            )
        graph_input: dict[str, JsonObject] = {
            "event": event.model_dump(mode="json"),
        }
    else:
        if "workflow" in saved_state:
            raise WorkflowCheckpointError(
                WorkflowCheckpointErrorCode.ALREADY_EXISTS,
                "工作流 checkpoint 已存在",
            )
        if workflow.workflow_id != event.workflow_id:
            # 在输入进入 checkpoint 前拒绝错配，避免持久化无效状态。
            raise WorkflowTransitionError(
                WorkflowTransitionErrorCode.WORKFLOW_MISMATCH,
                "工作流事件不属于当前工作流",
            )
        graph_input = {
            "workflow": workflow.model_dump(mode="json"),
            "event": event.model_dump(mode="json"),
        }

    result = graph.invoke(graph_input, config=config)
    return WorkflowState.model_validate(result["workflow"])


def workflow_checkpoint_config(workflow_id: UUID) -> RunnableConfig:
    """用业务工作流标识稳定映射 LangGraph thread。"""

    return {"configurable": {"thread_id": str(workflow_id)}}


def _apply_event(
    state: WorkflowGraphState,
    *,
    operation_plan_validator: OperationPlanValidator | None,
) -> dict[str, JsonObject]:
    """只委托给手写状态机，不复制业务规则或执行外部副作用。"""

    workflow = WorkflowState.model_validate(state["workflow"])
    event = WorkflowEvent.model_validate(state["event"])
    if event.kind == "workflow_completed":
        if operation_plan_validator is None:
            raise WorkflowBoundaryError(
                WorkflowBoundaryErrorCode.OPERATION_PLAN_VALIDATOR_REQUIRED,
                "完成工作流前必须经过操作计划验证 Service",
            )
        operation_plan_validator(workflow.operation_plan)

    next_workflow = transition_workflow(workflow, event)
    return {
        "workflow": next_workflow.model_dump(mode="json"),
    }
