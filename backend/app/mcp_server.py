"""FileNest 只读检索与待审批提案的 MCP 协议适配层。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import partial
from typing import Any, Literal

import anyio
import mcp.types as mcp_types
from langgraph.graph.state import CompiledStateGraph
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.orm import Session

from .database import SessionFactory
from .organization_planning import (
    CreateApprovalWorkflowRequest,
    create_waiting_approval_workflow,
)
from .path_policy import PathPolicyError
from .read_tools import build_knowledge_search_tool, build_search_files_tool
from .services import (
    FileEntryNotFoundError,
    OperationPlanExpiredError,
    OperationPlanSourceChangedError,
    OperationPlanSourceMismatchError,
    OperationPlanTargetConflictError,
    OperationPlanTargetUnavailableError,
    OperationPreviewPathUnavailableError,
    WorkspaceNotFoundError,
    validate_operation_plan,
)
from .tool_contracts import ToolResult, _safe_validation_errors
from .tool_registry import ToolRegistry
from .workflow import WorkflowState
from .workflow_graph import WorkflowCheckpointError, open_checkpointed_workflow_graph
from .workflow_runtime import WORKFLOW_CHECKPOINT_PATH


logger = logging.getLogger(__name__)

MCP_READ_TOOL_NAMES = ("search_files", "knowledge_search")
MCP_PROPOSAL_TOOL_NAME = "create_operation_proposal"
MCP_TOOL_NAMES = (*MCP_READ_TOOL_NAMES, MCP_PROPOSAL_TOOL_NAME)

WorkflowGraphFactory = Callable[
    [Session],
    AbstractContextManager[CompiledStateGraph],
]


class MCPOperationProposal(BaseModel):
    """MCP 返回的待审批提案；不提供任何执行结果或执行入口。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: int
    workflow: WorkflowState
    approval_status: Literal["WAITING_APPROVAL"] = "WAITING_APPROVAL"


def _default_workflow_graph_factory(
    session: Session,
) -> AbstractContextManager[CompiledStateGraph]:
    """复用正式 API 的 checkpoint 路径与计划校验边界。"""

    return open_checkpointed_workflow_graph(
        WORKFLOW_CHECKPOINT_PATH,
        operation_plan_validator=partial(validate_operation_plan, session),
    )


def build_mcp_read_tool_registry(session: Session) -> ToolRegistry:
    """构建 MCP 当前允许暴露的 FileNest 只读工具白名单。"""

    return ToolRegistry(
        [
            build_search_files_tool(session),
            build_knowledge_search_tool(session),
        ]
    )


class FileNestMCPServer:
    """把 MCP 请求接到既有只读工具与审批工作流边界。"""

    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionFactory,
        workflow_graph_factory: WorkflowGraphFactory = (
            _default_workflow_graph_factory
        ),
    ) -> None:
        self._session_factory = session_factory
        self._workflow_graph_factory = workflow_graph_factory
        self._server = Server(
            "filenest",
            version="1.0",
            description="FileNest controlled search and approval proposal server",
            instructions=(
                "Indexed file and knowledge searches are read-only evidence. "
                "Operation proposals only enter WAITING_APPROVAL; approval, execution, "
                "and undo remain outside this MCP server."
            ),
            on_list_tools=self._handle_list_tools,
            on_call_tool=self._handle_call_tool,
        )

    @property
    def server(self) -> Server:
        """返回官方 SDK server，供 stdio 运行入口使用。"""

        return self._server

    async def list_tools(self) -> mcp_types.ListToolsResult:
        """提供无传输的测试入口，使用与协议回调相同的实现。"""

        return await self._handle_list_tools(None, None)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> mcp_types.CallToolResult:
        """提供无传输的测试入口，使用与协议回调相同的实现。"""

        params = mcp_types.CallToolRequestParams(name=name, arguments=arguments)
        return await self._handle_call_tool(None, params)

    async def run_stdio(self) -> None:
        """在本机 stdin/stdout 上运行 MCP，暂不开放网络 transport。"""

        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    async def _handle_list_tools(
        self,
        _context: Any,
        _params: mcp_types.PaginatedRequestParams | None,
    ) -> mcp_types.ListToolsResult:
        with self._session_factory() as session:
            definitions = build_mcp_read_tool_registry(session).definitions()

        return mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(
                    name=definition.name,
                    description=definition.description,
                    input_schema=definition.parameters,
                    annotations=mcp_types.ToolAnnotations(
                        read_only_hint=True,
                        destructive_hint=False,
                        idempotent_hint=True,
                        open_world_hint=False,
                    ),
                )
                for definition in definitions
            ]
            + [_operation_proposal_tool()],
        )

    async def _handle_call_tool(
        self,
        _context: Any,
        params: mcp_types.CallToolRequestParams,
    ) -> mcp_types.CallToolResult:
        if params.name not in MCP_TOOL_NAMES:
            result = ToolResult.failure(
                code="unknown_tool",
                message="请求的工具未注册",
            )
        else:
            try:
                if params.name == MCP_PROPOSAL_TOOL_NAME:
                    result = await anyio.to_thread.run_sync(
                        self._invoke_operation_proposal,
                        params.arguments or {},
                    )
                else:
                    result = await anyio.to_thread.run_sync(
                        self._invoke_read_tool,
                        params.name,
                        params.arguments or {},
                    )
            except Exception:
                logger.exception("MCP tool invocation failed: %s", params.name)
                result = ToolResult.failure(
                    code="mcp_tool_unavailable",
                    message="MCP 工具当前不可用",
                    details={"tool": params.name},
                )

        return _to_mcp_result(result)

    def _invoke_read_tool(self, name: str, arguments: object) -> ToolResult:
        with self._session_factory() as session:
            return build_mcp_read_tool_registry(session).invoke(name, arguments)

    def _invoke_operation_proposal(self, arguments: object) -> ToolResult:
        """只调用现有工作流 Service 创建待审批状态，不触碰用户文件。"""

        try:
            request = CreateApprovalWorkflowRequest.model_validate(arguments)
        except ValidationError as error:
            return ToolResult.failure(
                code="invalid_arguments",
                message="工具参数不符合契约",
                details={"errors": _safe_validation_errors(error)},
            )

        try:
            with self._session_factory() as session:
                with self._workflow_graph_factory(session) as graph:
                    created = create_waiting_approval_workflow(
                        session,
                        graph,
                        request,
                    )
        except WorkspaceNotFoundError:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
            )
        except FileEntryNotFoundError:
            return ToolResult.failure(
                code="file_not_found",
                message="文件索引不存在",
            )
        except (
            OperationPreviewPathUnavailableError,
            OperationPlanExpiredError,
            OperationPlanSourceChangedError,
            OperationPlanSourceMismatchError,
            OperationPlanTargetConflictError,
            OperationPlanTargetUnavailableError,
            PathPolicyError,
        ):
            return ToolResult.failure(
                code="organization_plan_unavailable",
                message="当前文件状态无法生成安全计划",
            )
        except WorkflowCheckpointError as error:
            return ToolResult.failure(
                code=str(error.code),
                message="工作流 checkpoint 冲突",
            )

        proposal = MCPOperationProposal(
            approval_id=created.approval_id,
            workflow=created.workflow,
        )
        return ToolResult.success(proposal.model_dump(mode="json"))


def _to_mcp_result(result: ToolResult) -> mcp_types.CallToolResult:
    """保留 FileNest 结果信封，同时转换为 MCP 的结构化结果。"""

    payload = result.model_dump(mode="json")
    content = mcp_types.TextContent(
        text=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return mcp_types.CallToolResult(
        content=[content],
        structured_content=payload,
        is_error=not result.ok,
    )


def _operation_proposal_tool() -> mcp_types.Tool:
    """声明提案能力，但明确标注其不是只读工具，也不是执行工具。"""

    return mcp_types.Tool(
        name=MCP_PROPOSAL_TOOL_NAME,
        description=(
            "根据现有索引和实时文件状态创建待人工审批的操作提案；"
            "不会批准、执行或撤销文件操作。"
        ),
        input_schema=CreateApprovalWorkflowRequest.model_json_schema(),
        annotations=mcp_types.ToolAnnotations(
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )


def main() -> None:
    """运行本机 stdio MCP server。"""

    asyncio.run(FileNestMCPServer().run_stdio())


if __name__ == "__main__":
    main()
