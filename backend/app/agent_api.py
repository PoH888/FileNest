"""最小界面使用的同步只读 Agent HTTP 边界。"""

from collections.abc import Callable, Mapping
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy.orm import Session

from .agent_loop import AgentLoop, AgentRunStatus
from .agent_observability import (
    AgentObservabilityError,
    SqlAlchemyAgentRunRecorder,
)
from .database import get_session
from .model_client import ModelClient, ModelMessage
from .model_settings import ModelSettings
from .openai_compatible_model_client import (
    OpenAICompatibleModelClient,
    UnsupportedModelProviderError,
)
from .read_tools import build_get_file_metadata_tool, build_search_files_tool
from .services import get_workspace as get_workspace_service
from .tool_contracts import ToolResult
from .tool_registry import ToolRegistry


router = APIRouter(prefix="/api/v1")


class AgentRunRequest(BaseModel):
    """最小界面允许提交的一次工作区内只读请求。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    request_text: str = Field(min_length=1, max_length=2_000)

    @field_validator("request_text")
    @classmethod
    def normalize_request_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("request_text must not be blank")
        return normalized


class AgentRunResponseError(BaseModel):
    """可以安全交给界面显示或判断的模型请求错误。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    retryable: bool
    attempts: int = Field(ge=1)


class AgentSourceReference(BaseModel):
    """来自成功只读工具结果的一条文件出处。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: int = Field(ge=1)
    file_id: int = Field(ge=1)
    name: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)


class AgentRunResponse(BaseModel):
    """一次同步 Agent Run 的最小公开结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: int = Field(ge=1)
    status: AgentRunStatus
    final_answer: str | None = None
    error: AgentRunResponseError | None = None
    sources: tuple[AgentSourceReference, ...] = ()


class AgentRunExecutor(Protocol):
    """供 API 注入真实或确定性 Agent 执行器。"""

    def run(
        self,
        session: Session,
        *,
        workspace_id: int,
        request_text: str,
    ) -> AgentRunResponse: ...


class _ModelConfigurationUnavailableError(RuntimeError):
    """把配置校验失败收敛为不含密钥和原始输入的内部错误。"""


class _WorkspaceScopedToolRegistry(ToolRegistry):
    """只注册只读工具，并拒绝模型改用其他工作区。"""

    def __init__(self, session: Session, workspace_id: int) -> None:
        super().__init__(
            [
                build_search_files_tool(session),
                build_get_file_metadata_tool(session),
            ]
        )
        self._workspace_id = workspace_id

    def validate(self, name: object, arguments: object) -> ToolResult:
        scope_error = self._scope_error(name, arguments)
        if scope_error is not None:
            return scope_error
        return super().validate(name, arguments)

    def invoke(self, name: object, arguments: object) -> ToolResult:
        scope_error = self._scope_error(name, arguments)
        if scope_error is not None:
            return scope_error
        return super().invoke(name, arguments)

    def _scope_error(
        self,
        name: object,
        arguments: object,
    ) -> ToolResult | None:
        if name not in self.names or not isinstance(arguments, Mapping):
            return None
        requested_workspace_id = arguments.get("workspace_id")
        if requested_workspace_id == self._workspace_id:
            return None
        return ToolResult.failure(
            code="invalid_arguments",
            message="请求的工作区不在当前 Agent Run 授权范围内",
        )


class _CapturingAgentRunRecorder(SqlAlchemyAgentRunRecorder):
    """保留现有安全记录行为，同时把本次 run_id 交给 API。"""

    run_id: int | None = None

    def start_run(self) -> int:
        self.run_id = super().start_run()
        return self.run_id


class ReadOnlyAgentRunExecutor:
    """使用真实模型客户端和工作区受限只读工具执行请求。"""

    def __init__(
        self,
        model_client_factory: Callable[[], ModelClient] | None = None,
    ) -> None:
        self._model_client_factory = model_client_factory or _build_model_client

    def run(
        self,
        session: Session,
        *,
        workspace_id: int,
        request_text: str,
    ) -> AgentRunResponse:
        recorder = _CapturingAgentRunRecorder(session)
        loop = AgentLoop(
            model_client=self._model_client_factory(),
            tool_registry=_WorkspaceScopedToolRegistry(session, workspace_id),
            recorder=recorder,
        )
        result = loop.run(
            [
                ModelMessage(
                    role="system",
                    content=(
                        "你是 FileNest 只读文件查询助手。"
                        f"本次只允许查询工作区 {workspace_id}，"
                        "不得请求写工具或其他工作区。"
                    ),
                ),
                ModelMessage(role="user", content=request_text),
            ]
        )
        if recorder.run_id is None:
            raise AgentObservabilityError("Agent 运行记录不存在")

        response_error = (
            AgentRunResponseError(
                code=result.error.code,
                retryable=result.error.retryable,
                attempts=result.error.attempts,
            )
            if result.error is not None
            else None
        )
        return AgentRunResponse(
            run_id=recorder.run_id,
            status=result.status,
            final_answer=result.final_answer,
            error=response_error,
            sources=_source_references(result.messages, workspace_id),
        )


def _build_model_client() -> ModelClient:
    try:
        return OpenAICompatibleModelClient(ModelSettings())
    except (ValidationError, UnsupportedModelProviderError) as error:
        raise _ModelConfigurationUnavailableError from error


def _source_references(
    messages: tuple[ModelMessage, ...],
    workspace_id: int,
) -> tuple[AgentSourceReference, ...]:
    """只接受 Agent Loop 生成的成功只读工具消息作为出处证据。"""

    tool_names = {
        tool_call.id: tool_call.name
        for message in messages
        if message.role == "assistant"
        for tool_call in message.tool_calls
    }
    references: list[AgentSourceReference] = []
    seen_file_ids: set[int] = set()

    for message in messages:
        if message.role != "tool" or message.content is None:
            continue
        tool_name = tool_names.get(message.tool_call_id or "")
        if tool_name not in {"search_files", "get_file_metadata"}:
            continue
        try:
            tool_result = ToolResult.model_validate_json(message.content)
        except ValidationError:
            continue
        if not tool_result.ok or not isinstance(tool_result.data, dict):
            continue

        raw_items: object
        if tool_name == "search_files":
            raw_items = tool_result.data.get("items", [])
        else:
            result_workspace_id = tool_result.data.get("workspace_id")
            raw_items = (
                [tool_result.data]
                if result_workspace_id == workspace_id
                else []
            )
        if not isinstance(raw_items, list):
            continue

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            try:
                reference = AgentSourceReference(
                    workspace_id=workspace_id,
                    file_id=raw_item.get("file_id"),
                    name=raw_item.get("name"),
                    relative_path=raw_item.get("relative_path"),
                )
            except ValidationError:
                continue
            if reference.file_id in seen_file_ids:
                continue
            references.append(reference)
            seen_file_ids.add(reference.file_id)

    return tuple(references)


_default_executor = ReadOnlyAgentRunExecutor()


def get_agent_run_executor() -> AgentRunExecutor:
    """返回延迟读取模型环境配置的正式执行器。"""

    return _default_executor


@router.post("/agent-runs", response_model=AgentRunResponse)
def create_agent_run(
    request: AgentRunRequest,
    session: Session = Depends(get_session),
    executor: AgentRunExecutor = Depends(get_agent_run_executor),
) -> AgentRunResponse:
    """在已登记工作区内执行一次同步只读 Agent 请求。"""

    if get_workspace_service(session, request.workspace_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "workspace_not_found",
                "message": "工作区不存在。",
            },
        )

    try:
        return executor.run(
            session,
            workspace_id=request.workspace_id,
            request_text=request.request_text,
        )
    except _ModelConfigurationUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_not_configured",
                "message": "模型配置当前不可用。",
            },
        ) from error
    except AgentObservabilityError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "agent_run_unavailable",
                "message": "Agent 运行当前不可用。",
            },
        ) from error
