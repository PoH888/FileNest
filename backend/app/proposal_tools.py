"""Agent 可调用的文件操作 Proposal 工具。"""

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast
from uuid import UUID, uuid4

from langgraph.graph.state import CompiledStateGraph
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from sqlalchemy.orm import Session

from .filesystem_adapter import FileSystemAdapter
from .operation_preview import OperationPreviewRequest
from .operation_plan import (
    ContentHash,
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from .organization_planning import (
    CreateApprovalWorkflowRequest,
    OrganizationTargetSelection,
    create_waiting_approval_workflow,
    create_waiting_approval_workflow_for_plan,
)
from .path_policy import PathPolicyError
from .quarantine import (
    QuarantineError,
    QuarantineManager,
    build_quarantine_relative_path,
)
from .services import (
    FileEntryNotFoundError,
    OperationPlanExpiredError,
    OperationPlanSourceChangedError,
    OperationPlanSourceMismatchError,
    OperationPlanTargetConflictError,
    OperationPlanTargetUnavailableError,
    OperationPreviewPathUnavailableError,
    WorkspaceNotFoundError,
    get_authorized_file_metadata,
    get_workspace,
    validate_operation_plan,
)
from .tool_contracts import Tool, ToolResult
from .workflow_graph import WorkflowCheckpointError


class ProposeMoveArguments(BaseModel):
    """提出文件移动 Proposal 时允许 Agent 提供的参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    source_file_id: int = Field(ge=1)
    destination: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_destination(self) -> "ProposeMoveArguments":
        """复用整理预览的路径契约，先拒绝越界或未规范化目标。"""

        OperationPreviewRequest(
            workspace_id=self.workspace_id,
            source_file_ids=(self.source_file_id,),
            target_directories=(self.destination,),
        )
        return self


class ProposeMoveData(BaseModel):
    """移动 Proposal 成功后返回给 Agent 的最小结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: UUID


class ProposeRenameArguments(BaseModel):
    """提出文件重命名 Proposal 时允许 Agent 提供的参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    source_file_id: int = Field(ge=1)
    new_name: str = Field(min_length=1, max_length=255)

    @field_validator("new_name")
    @classmethod
    def validate_new_name(cls, value: str) -> str:
        """只接受单个文件名，避免把重命名变成路径移动。"""

        if value != value.strip():
            raise ValueError("new_name must not have surrounding whitespace")
        if value in {".", ".."}:
            raise ValueError("new_name must be a file name")
        if (
            "/" in value
            or "\\" in value
            or "\x00" in value
            or PureWindowsPath(value).drive
        ):
            raise ValueError("new_name must not contain a path")
        return value


class ProposeRenameData(BaseModel):
    """重命名 Proposal 成功后返回的待审批计划标识。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: UUID


class ProposeQuarantineArguments(BaseModel):
    """提出文件隔离 Proposal 时允许 Agent 提供的参数。"""

    model_config = ConfigDict(extra="forbid")

    workspace_id: int = Field(ge=1)
    source_file_id: int = Field(ge=1)


class ProposeQuarantineData(BaseModel):
    """隔离 Proposal 成功后返回的计划标识和确定目标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: UUID
    quarantine_destination: str = Field(min_length=1)


_PLAN_VALIDATION_ERRORS = (
    OperationPlanExpiredError,
    OperationPlanSourceChangedError,
    OperationPlanSourceMismatchError,
    OperationPlanTargetConflictError,
    OperationPlanTargetUnavailableError,
    OperationPreviewPathUnavailableError,
    PathPolicyError,
)


def build_propose_move_tool(
    session: Session,
    graph: CompiledStateGraph,
    *,
    workflow_id_factory: Callable[[], UUID] = uuid4,
    plan_id_factory: Callable[[], UUID] = uuid4,
) -> Tool:
    """构建只创建待审批移动计划、不会写入文件系统的 Agent 工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(ProposeMoveArguments, arguments)
        request = CreateApprovalWorkflowRequest(
            workspace_id=options.workspace_id,
            target_directories=(options.destination,),
            selections=(
                OrganizationTargetSelection(
                    source_file_id=options.source_file_id,
                    target_directory=options.destination,
                ),
            ),
        )

        try:
            created = create_waiting_approval_workflow(
                session,
                graph,
                request,
                workflow_id_factory=workflow_id_factory,
                plan_id_factory=plan_id_factory,
            )
        except WorkspaceNotFoundError:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
                details={"workspace_id": options.workspace_id},
            )
        except FileEntryNotFoundError:
            return ToolResult.failure(
                code="file_not_found",
                message="文件索引不存在或不属于当前工作区",
                details={
                    "workspace_id": options.workspace_id,
                    "source_file_id": options.source_file_id,
                },
            )
        except _PLAN_VALIDATION_ERRORS:
            return ToolResult.failure(
                code="proposal_unavailable",
                message="当前文件状态无法生成安全移动计划",
            )
        except WorkflowCheckpointError:
            return ToolResult.failure(
                code="proposal_unavailable",
                message="移动计划当前无法进入待审批状态",
            )

        return ToolResult.success(
            ProposeMoveData(
                plan_id=created.workflow.operation_plan.plan_id,
            ).model_dump(mode="json")
        )

    return Tool(
        name="propose_move",
        description=(
            "为指定工作区内的文件创建待人工审批的移动计划；"
            "不会直接写入或移动文件。"
        ),
        arguments_model=ProposeMoveArguments,
        handler=handle,
    )


def build_propose_rename_tool(
    session: Session,
    graph: CompiledStateGraph,
    *,
    workflow_id_factory: Callable[[], UUID] = uuid4,
    plan_id_factory: Callable[[], UUID] = uuid4,
) -> Tool:
    """构建只创建待审批重命名计划、不会写入文件系统的 Agent 工具。"""

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(ProposeRenameArguments, arguments)

        try:
            workspace = get_workspace(session, options.workspace_id)
            if workspace is None:
                return ToolResult.failure(
                    code="workspace_not_found",
                    message="工作区不存在",
                    details={"workspace_id": options.workspace_id},
                )

            file_entry = get_authorized_file_metadata(
                session,
                options.workspace_id,
                options.source_file_id,
            )
            adapter = FileSystemAdapter(Path(workspace.root_path))
            source_path = Path(file_entry.relative_path)
            try:
                source_metadata = adapter.get_file_metadata(source_path)
                source_hash = adapter.get_file_sha256(source_path)
                if source_metadata is None or source_hash is None:
                    return ToolResult.failure(
                        code="source_unavailable",
                        message="源文件当前不可用",
                    )
            except OSError:
                return ToolResult.failure(
                    code="source_unavailable",
                    message="源文件当前不可用",
                )

            source_relative_path = PurePosixPath(file_entry.relative_path)
            target_relative_path = (
                source_relative_path.parent / options.new_name
            ).as_posix()
            target_path = Path(target_relative_path)
            adapter.authorized_path(target_path)

            if target_relative_path == file_entry.relative_path:
                return ToolResult.failure(
                    code="same_path",
                    message="新名称与原文件名称相同",
                )
            if adapter.path_exists(target_path):
                return ToolResult.failure(
                    code="target_conflict",
                    message="目标文件已存在",
                )

            created_at = datetime.now(timezone.utc)
            plan = OperationPlan(
                plan_id=plan_id_factory(),
                workspace_id=options.workspace_id,
                created_at=created_at,
                operations=(
                    OperationPlanItem(
                        operation_type="rename",
                        source_file_id=file_entry.id,
                        source_relative_path=file_entry.relative_path,
                        target_relative_path=target_relative_path,
                        source_precondition=FilePrecondition(
                            size_bytes=source_metadata.size_bytes,
                            mtime_ns=source_metadata.mtime_ns,
                            content_hash=ContentHash(digest=source_hash),
                        ),
                        reason=OperationReason(
                            kind="manual_selection",
                            description="Agent 提出的重命名操作",
                        ),
                    ),
                ),
            )
            validate_operation_plan(session, plan, now=created_at)
            created = create_waiting_approval_workflow_for_plan(
                session,
                graph,
                plan,
                workflow_id_factory=workflow_id_factory,
            )
        except WorkspaceNotFoundError:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
                details={"workspace_id": options.workspace_id},
            )
        except FileEntryNotFoundError:
            return ToolResult.failure(
                code="file_not_found",
                message="文件索引不存在或不属于当前工作区",
                details={
                    "workspace_id": options.workspace_id,
                    "source_file_id": options.source_file_id,
                },
            )
        except PathPolicyError as error:
            return ToolResult.failure(
                code=error.code.value,
                message=error.message,
            )
        except _PLAN_VALIDATION_ERRORS:
            return ToolResult.failure(
                code="proposal_unavailable",
                message="当前文件状态无法生成安全重命名计划",
            )
        except WorkflowCheckpointError:
            return ToolResult.failure(
                code="proposal_unavailable",
                message="重命名计划当前无法进入待审批状态",
            )

        return ToolResult.success(
            ProposeRenameData(
                plan_id=created.workflow.operation_plan.plan_id,
            ).model_dump(mode="json")
        )

    return Tool(
        name="propose_rename",
        description=(
            "为指定工作区内的文件创建重命名 Proposal；"
            "只校验目标路径，不实际执行 rename。"
        ),
        arguments_model=ProposeRenameArguments,
        handler=handle,
    )


def build_propose_quarantine_tool(
    session: Session,
    graph: CompiledStateGraph,
    *,
    quarantine_root: Path,
    workflow_id_factory: Callable[[], UUID] = uuid4,
    plan_id_factory: Callable[[], UUID] = uuid4,
) -> Tool:
    """构建只创建隔离计划、不会直接移动文件的 Agent 工具。"""

    quarantine_adapter = FileSystemAdapter(Path(quarantine_root))

    def handle(arguments: BaseModel) -> ToolResult:
        options = cast(ProposeQuarantineArguments, arguments)

        try:
            workspace = get_workspace(session, options.workspace_id)
            if workspace is None:
                return ToolResult.failure(
                    code="workspace_not_found",
                    message="工作区不存在",
                    details={"workspace_id": options.workspace_id},
                )

            file_entry = get_authorized_file_metadata(
                session,
                options.workspace_id,
                options.source_file_id,
            )
            workspace_adapter = FileSystemAdapter(Path(workspace.root_path))
            # 构造管理器只校验两个根目录互不包含，不调用会产生移动副作用的方法。
            QuarantineManager(workspace_adapter, quarantine_adapter)

            source_path = Path(file_entry.relative_path)
            try:
                source_metadata = workspace_adapter.get_file_metadata(source_path)
                source_hash = workspace_adapter.get_file_sha256(source_path)
            except OSError:
                return ToolResult.failure(
                    code="quarantine_source_unavailable",
                    message="待隔离的源文件当前不可用",
                )
            if source_metadata is None or source_hash is None:
                return ToolResult.failure(
                    code="quarantine_source_unavailable",
                    message="待隔离的源文件当前不可用",
                )

            plan_id = plan_id_factory()
            target_relative_path = build_quarantine_relative_path(
                workspace_id=options.workspace_id,
                plan_id=plan_id,
                source_file_id=options.source_file_id,
                file_name=source_path.name,
            )
            target_path = quarantine_adapter.authorized_path(
                target_relative_path
            )
            if quarantine_adapter.path_exists(target_path):
                return ToolResult.failure(
                    code="quarantine_target_conflict",
                    message="隔离目标已经存在，禁止覆盖",
                )

            plan = OperationPlan(
                plan_id=plan_id,
                workspace_id=options.workspace_id,
                created_at=datetime.now(timezone.utc),
                operations=(
                    OperationPlanItem(
                        operation_type="quarantine",
                        source_file_id=file_entry.id,
                        source_relative_path=file_entry.relative_path,
                        target_relative_path=target_relative_path.as_posix(),
                        source_precondition=FilePrecondition(
                            size_bytes=source_metadata.size_bytes,
                            mtime_ns=source_metadata.mtime_ns,
                            content_hash=ContentHash(digest=source_hash),
                        ),
                        reason=OperationReason(
                            kind="manual_selection",
                            description="Agent 提出的隔离操作",
                        ),
                    ),
                ),
            )
            create_waiting_approval_workflow_for_plan(
                session,
                graph,
                plan,
                workflow_id_factory=workflow_id_factory,
            )
        except WorkspaceNotFoundError:
            return ToolResult.failure(
                code="workspace_not_found",
                message="工作区不存在",
                details={"workspace_id": options.workspace_id},
            )
        except FileEntryNotFoundError:
            return ToolResult.failure(
                code="file_not_found",
                message="文件索引不存在或不属于当前工作区",
                details={
                    "workspace_id": options.workspace_id,
                    "source_file_id": options.source_file_id,
                },
            )
        except (PathPolicyError, QuarantineError) as error:
            return ToolResult.failure(
                code=error.code.value,
                message=(
                    error.message
                    if isinstance(error, PathPolicyError)
                    else str(error)
                ),
            )
        except WorkflowCheckpointError:
            return ToolResult.failure(
                code="proposal_unavailable",
                message="隔离计划当前无法进入待审批状态",
            )

        return ToolResult.success(
            ProposeQuarantineData(
                plan_id=plan.plan_id,
                quarantine_destination=target_relative_path.as_posix(),
            ).model_dump(mode="json")
        )

    return Tool(
        name="propose_quarantine",
        description=(
            "为指定工作区内的文件创建待人工审批的隔离计划；"
            "不会直接移动文件。"
        ),
        arguments_model=ProposeQuarantineArguments,
        handler=handle,
    )
