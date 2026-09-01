"""从只读预览确定性创建待人工审批的整理工作流。"""

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from .filesystem_adapter import FileSystemAdapter
from .models import (
    ApprovalRequest,
    OperationItemRecord,
    OperationPlanRecord,
    OperationStatusRecord,
)
from .operation_plan import (
    ContentHash,
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
)
from .operation_preview import OperationPreviewRequest
from .operation_status import OperationStatus
from .path_policy import (
    DEFAULT_WORKSPACE_POLICY,
    WorkspacePolicy,
    serialize_workspace_policy_rules,
)
from .repositories import (
    add_approval_request,
    add_operation_plan,
    add_operation_status,
    compare_and_set_operation_status,
)
from .services import (
    OperationPreviewPathUnavailableError,
    generate_operation_preview,
    get_authorized_file_metadata,
    get_workspace,
    require_workspace_proposal_policy,
    validate_operation_plan,
)
from .workflow import WorkflowEvent, WorkflowState
from .workflow_graph import run_checkpointed_workflow_event


UuidFactory = Callable[[], UUID]


class OrganizationTargetSelection(BaseModel):
    """用户为一个源文件选择的已提供目标目录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_file_id: int = Field(ge=1)
    target_directory: str = Field(min_length=1)


class CreateApprovalWorkflowRequest(BaseModel):
    """重新生成预览并创建确定计划所需的最小输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: int = Field(ge=1)
    target_directories: tuple[str, ...] = Field(min_length=1, max_length=200)
    selections: tuple[OrganizationTargetSelection, ...] = Field(
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_preview_inputs(self) -> "CreateApprovalWorkflowRequest":
        source_file_ids = tuple(
            selection.source_file_id for selection in self.selections
        )
        OperationPreviewRequest(
            workspace_id=self.workspace_id,
            source_file_ids=source_file_ids,
            target_directories=self.target_directories,
        )
        offered_targets = set(self.target_directories)
        if any(
            selection.target_directory not in offered_targets
            for selection in self.selections
        ):
            raise ValueError(
                "every selected target must be present in target_directories"
            )
        return self


class EditOrganizationPlanRequest(BaseModel):
    """用户对当前计划提交的最小目标目录修改。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    changes: tuple[OrganizationTargetSelection, ...] = Field(
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "EditOrganizationPlanRequest":
        source_file_ids = tuple(change.source_file_id for change in self.changes)
        if len(set(source_file_ids)) != len(source_file_ids):
            raise ValueError("changes must contain unique source_file_ids")
        return self


class CreatedApprovalWorkflow(BaseModel):
    """已经同时通过计划校验、checkpoint 和审批持久化的结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: int = Field(ge=1)
    workflow: WorkflowState


def build_operation_plan_record(
    plan: OperationPlan,
    *,
    workflow_id: UUID,
    agent_run_id: int | None = None,
    parent_plan_id: UUID | None = None,
    policy: WorkspacePolicy | None = None,
) -> OperationPlanRecord:
    """将已验证的计划契约转换为可在业务库中保存的完整记录。"""

    effective_policy = policy or DEFAULT_WORKSPACE_POLICY
    user_denylist, ignore_patterns = serialize_workspace_policy_rules(
        effective_policy
    )
    return OperationPlanRecord(
        plan_id=str(plan.plan_id),
        schema_version=plan.schema_version,
        workspace_id=plan.workspace_id,
        agent_run_id=agent_run_id,
        workflow_id=str(workflow_id),
        operation_type=plan.operations[0].operation_type,
        metadata_json=json.dumps(
            {
                "contract_schema_version": plan.schema_version,
                "workspace_policy": {
                    "policy_revision": effective_policy.policy_revision,
                    "read_enabled": effective_policy.read_enabled,
                    "proposal_enabled": effective_policy.proposal_enabled,
                    "safe_execution_enabled": (
                        effective_policy.safe_execution_enabled
                    ),
                    "user_denylist_json": user_denylist,
                    "ignore_patterns_json": ignore_patterns,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        status="WAITING_APPROVAL",
        parent_plan_id=(
            str(parent_plan_id) if parent_plan_id is not None else None
        ),
        created_at=plan.created_at,
        items=[
            OperationItemRecord(
                sequence_no=sequence_no,
                operation_type=operation.operation_type,
                source_file_id=operation.source_file_id,
                source_relative_path=operation.source_relative_path,
                target_relative_path=operation.target_relative_path,
                source_size_bytes=operation.source_precondition.size_bytes,
                source_mtime_ns=operation.source_precondition.mtime_ns,
                source_hash_algorithm=(
                    operation.source_precondition.content_hash.algorithm
                    if operation.source_precondition.content_hash is not None
                    else None
                ),
                source_sha256=(
                    operation.source_precondition.content_hash.digest
                    if operation.source_precondition.content_hash is not None
                    else None
                ),
                reason_kind=operation.reason.kind,
                reason_description=operation.reason.description,
                reason_match_score=operation.reason.match_score,
                risks_json=json.dumps(
                    [risk.model_dump(mode="json") for risk in operation.risks],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                status="PENDING",
            )
            for sequence_no, operation in enumerate(plan.operations, start=1)
        ],
    )


def merge_edit_request(
    current_plan: OperationPlan,
    request: EditOrganizationPlanRequest,
) -> CreateApprovalWorkflowRequest:
    """把最小目标修改合并为后端可重新验证的完整计划请求。"""

    current_operations = {
        operation.source_file_id: operation
        for operation in current_plan.operations
    }
    changed_targets = {
        change.source_file_id: change.target_directory
        for change in request.changes
    }
    unknown_sources = set(changed_targets) - set(current_operations)
    if unknown_sources:
        raise ValueError("changes must reference files in the current plan")

    selections = tuple(
        OrganizationTargetSelection(
            source_file_id=operation.source_file_id,
            target_directory=changed_targets.get(
                operation.source_file_id,
                PurePosixPath(operation.target_relative_path).parent.as_posix(),
            ),
        )
        for operation in current_plan.operations
    )
    target_directories = tuple(
        dict.fromkeys(selection.target_directory for selection in selections)
    )
    return CreateApprovalWorkflowRequest(
        workspace_id=current_plan.workspace_id,
        target_directories=target_directories,
        selections=selections,
    )


def build_organization_plan(
    session: Session,
    request: CreateApprovalWorkflowRequest,
    *,
    now: datetime | None = None,
    plan_id_factory: UuidFactory = uuid4,
) -> OperationPlan:
    """从当前授权文件状态构建并校验完整的确定计划。"""

    created_at = now or datetime.now(timezone.utc)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("now must include timezone information")

    policy = require_workspace_proposal_policy(session, request.workspace_id)
    preview_request = OperationPreviewRequest(
        workspace_id=request.workspace_id,
        source_file_ids=tuple(
            selection.source_file_id for selection in request.selections
        ),
        target_directories=request.target_directories,
    )
    preview = generate_operation_preview(session, preview_request)
    workspace = get_workspace(session, request.workspace_id)
    if workspace is None:
        # generate_operation_preview 已保证该分支不可达，保留失败关闭边界。
        raise RuntimeError("validated workspace disappeared")

    adapter = FileSystemAdapter(
        Path(workspace.root_path),
        workspace_policy=policy,
    )
    preview_items = {item.source_file_id: item for item in preview.items}
    operations: list[OperationPlanItem] = []
    for selection in request.selections:
        file_entry = get_authorized_file_metadata(
            session,
            request.workspace_id,
            selection.source_file_id,
        )
        source_path = Path(file_entry.relative_path)
        try:
            metadata = adapter.get_file_metadata(source_path)
            content_hash = adapter.get_file_sha256(source_path)
        except OSError as error:
            raise OperationPreviewPathUnavailableError(
                file_entry.relative_path
            ) from error
        if metadata is None or content_hash is None:
            raise OperationPreviewPathUnavailableError(file_entry.relative_path)

        candidate_scores = {
            candidate.relative_directory: candidate.score
            for candidate in preview_items[file_entry.id].candidates
        }
        match_score = candidate_scores.get(selection.target_directory)
        reason = (
            OperationReason(
                kind="matched_candidate",
                description="采用只读预览中的候选目录",
                match_score=match_score,
            )
            if match_score is not None
            else OperationReason(
                kind="manual_selection",
                description="采用用户确认的已验证目标目录",
            )
        )
        target_relative_path = (
            PurePosixPath(selection.target_directory) / file_entry.name
        ).as_posix()
        operations.append(
            OperationPlanItem(
                source_file_id=file_entry.id,
                source_relative_path=file_entry.relative_path,
                target_relative_path=target_relative_path,
                source_precondition=FilePrecondition(
                    size_bytes=metadata.size_bytes,
                    mtime_ns=metadata.mtime_ns,
                    content_hash=ContentHash(digest=content_hash),
                ),
                reason=reason,
            )
        )

    plan = OperationPlan(
        plan_id=plan_id_factory(),
        workspace_id=request.workspace_id,
        created_at=created_at,
        operations=tuple(operations),
    )
    validate_operation_plan(session, plan, now=created_at)
    return plan


def create_waiting_approval_workflow(
    session: Session,
    graph: CompiledStateGraph,
    request: CreateApprovalWorkflowRequest,
    *,
    now: datetime | None = None,
    workflow_id_factory: UuidFactory = uuid4,
    plan_id_factory: UuidFactory = uuid4,
    agent_run_id: int | None = None,
) -> CreatedApprovalWorkflow:
    """安全构造计划，并在 checkpoint 成功后提交待审批业务状态。"""

    plan = build_organization_plan(
        session,
        request,
        now=now,
        plan_id_factory=plan_id_factory,
    )

    return create_waiting_approval_workflow_for_plan(
        session,
        graph,
        plan,
        workflow_id_factory=workflow_id_factory,
        agent_run_id=agent_run_id,
    )


def create_waiting_approval_workflow_for_plan(
    session: Session,
    graph: CompiledStateGraph,
    plan: OperationPlan,
    *,
    workflow_id_factory: UuidFactory = uuid4,
    agent_run_id: int | None = None,
) -> CreatedApprovalWorkflow:
    """把已完成业务校验的计划安全提交为待审批工作流。"""

    policy = require_workspace_proposal_policy(session, plan.workspace_id)
    workflow_id = workflow_id_factory()
    initial_workflow = WorkflowState(
        workflow_id=workflow_id,
        operation_plan=plan,
    )
    pause_event = WorkflowEvent(
        workflow_id=workflow_id,
        sequence_no=1,
        kind="pause_requested",
        reason_code="human_approval_required",
    )
    approval = ApprovalRequest(
        workflow_id=str(workflow_id),
        plan_id=str(plan.plan_id),
    )
    plan_record = build_operation_plan_record(
        plan,
        workflow_id=workflow_id,
        agent_run_id=agent_run_id,
        policy=policy,
    )

    try:
        # 先把完整业务计划写入主库；checkpoint 只是工作流状态投影。
        add_operation_plan(session, plan_record)
        session.flush()
        add_approval_request(session, approval)
        # 在 checkpoint 成功前不提交审批和计划事务。
        session.flush()
        operation_status = OperationStatusRecord(
            workflow_id=str(workflow_id),
            plan_id=str(plan.plan_id),
            approval_id=approval.id,
            overall_status=OperationStatus.PROPOSED.value,
        )
        add_operation_status(session, operation_status)
        session.flush()
        waiting_workflow = run_checkpointed_workflow_event(
            graph,
            pause_event,
            workflow=initial_workflow,
        )
        if (
            waiting_workflow.workflow_id != workflow_id
            or waiting_workflow.operation_plan.plan_id != plan.plan_id
            or waiting_workflow.status != "waiting"
            or waiting_workflow.wait_reason_code != "human_approval_required"
        ):
            raise RuntimeError(
                "checkpoint did not persist the expected approval state"
            )
        if not compare_and_set_operation_status(
            session,
            str(workflow_id),
            OperationStatus.PROPOSED,
            expected_revision=0,
            next_status=OperationStatus.WAITING_APPROVAL,
            plan_id=str(plan.plan_id),
            approval_id=approval.id,
        ):
            raise RuntimeError(
                "operation status did not enter the expected approval state"
            )
        session.expire(operation_status)
        approval_id = approval.id
        session.commit()
    except Exception:
        session.rollback()
        raise

    return CreatedApprovalWorkflow(
        approval_id=approval_id,
        workflow=waiting_workflow,
    )
