# FastAPI → Service → Repository → Session → SQLite

"""工作区业务服务层。

负责组织完整的工作区业务流程和事务，
不处理 HTTP 路由、状态码或响应格式。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import json
import logging
import os
from pathlib import Path
from typing import cast
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .filesystem_adapter import FileSystemAdapter
from .models import (
    ApprovalAuditEvent,
    ApprovalRequest,
    FileEntry,
    OperationItemRecord,
    OperationPlanRecord,
    Workspace,
    WorkspacePolicyAuditEvent,
)
from .operation_preview import (
    OperationPreviewItem,
    OperationPreviewRequest,
    OperationPreviewResponse,
    rank_preview_candidates,
)
from .operation_plan import (
    ContentHash,
    FilePrecondition,
    OperationPlan,
    OperationPlanItem,
    OperationReason,
    OperationRisk,
)
from .path_policy import (
    PathPolicyError,
    WorkspacePolicy,
    WorkspacePolicyPersistenceError,
    normalize_workspace_root,
    parse_workspace_policy,
    workspace_policy_rule_summary,
    validate_workspace_root,
)
from .quarantine import (
    QuarantineError,
    QuarantineManager,
    build_quarantine_relative_path,
    resolve_quarantine_root,
)
from .repositories import (
    add_file_entry,
    add_approval_audit_event,
    add_workspace_policy_audit_event,
    add_workspace,
    ApprovalAction,
    ApprovalStatus,
    compare_and_set_operation_plan_status,
    compare_and_set_approval_request,
    compare_and_set_operation_status,
    compare_and_set_operation_status_links,
    delete_file_entry,
    FileEntrySortField,
    find_file_entries,
    find_operation_plan_history,
    find_operation_plan_items,
    find_workspaces,
    get_file_entry_by_id,
    get_approval_request_by_workflow_id,
    get_operation_plan_by_id,
    get_operation_status_by_workflow_id,
    get_workspace_by_id,
    get_workspace_policy,
    get_workspace_policy_record,
    compare_and_set_workspace_policy,
    OperationPlanStatus,
    SortOrder,
)
from .operation_status import OperationStatus
from .workspace_scanner import IgnoredEntry, ScannedFile, scan_workspace_files


class WorkspacePathConflictError(Exception):
    """工作区根路径已经存在。"""


class WorkspaceNotFoundError(Exception):
    """文件索引操作所需的工作区不存在。"""


class WorkspacePolicyErrorCode(StrEnum):
    """Workspace Policy 服务层的稳定错误码。"""

    NOT_FOUND = "workspace_policy_not_found"
    INVALID = "workspace_policy_invalid"
    REVISION_CONFLICT = "workspace_policy_revision_conflict"
    AUDIT_WRITE_FAILED = "workspace_policy_audit_write_failed"
    READ_DISABLED = "workspace_policy_read_disabled"
    PROPOSAL_DISABLED = "workspace_policy_proposal_disabled"


class WorkspacePolicyError(RuntimeError):
    """拒绝缺失、损坏或并发过期的 Workspace Policy 变更。"""

    def __init__(
        self,
        code: WorkspacePolicyErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class FileEntryNotFoundError(Exception):
    """指定工作区内不存在所需的文件索引。"""


class DuplicateScannedPathError(Exception):
    """一次扫描结果中出现重复的工作区相对路径。"""


class WorkspaceScanUnavailableError(Exception):
    """工作区根目录当前无法安全扫描。"""


class OperationPreviewPathUnavailableError(Exception):
    """整理预览所需的源文件或目标目录当前不可用。"""


class OperationPlanSourceMismatchError(Exception):
    """计划中的源路径与当前工作区文件索引不一致。"""


class OperationPlanTargetUnavailableError(Exception):
    """计划目标的父目录当前不可用。"""


class OperationPlanTargetConflictError(Exception):
    """计划目标已经被文件、目录或符号链接占用。"""


class OperationPlanExpiredError(Exception):
    """计划的生成时间不在允许执行的时间窗口内。"""


class OperationPlanSourceChangedError(Exception):
    """计划生成后源文件已经消失或内容状态发生变化。"""


class OperationPlanPersistenceError(RuntimeError):
    """持久化计划无法安全还原为已验证的业务契约。"""


class ApprovalTransitionErrorCode(StrEnum):
    """审批转换失败时供程序稳定判断的错误码。"""

    NOT_FOUND = "approval_request_not_found"
    NOT_WAITING = "approval_not_waiting"
    PLAN_MISMATCH = "approval_plan_mismatch"
    PLAN_UNCHANGED = "approval_plan_unchanged"
    STATE_CHANGED = "approval_state_changed"


class ApprovalTransitionError(RuntimeError):
    """拒绝缺失、过期或不符合审批状态机规则的决定。"""

    def __init__(
        self,
        code: ApprovalTransitionErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


class OperationPlanApprovalErrorCode(StrEnum):
    """操作计划未通过人工审批时供程序稳定判断的错误码。"""

    NOT_FOUND = "operation_plan_approval_not_found"
    NOT_APPROVED = "operation_plan_not_approved"
    PLAN_MISMATCH = "approved_operation_plan_mismatch"


class OperationPlanApprovalError(RuntimeError):
    """阻止缺少有效人工审批的操作计划进入执行边界。"""

    def __init__(
        self,
        code: OperationPlanApprovalErrorCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class FileIndexSyncResult:
    """一次文件索引同步产生的变化统计。"""

    created: int
    updated: int
    deleted: int
    unchanged: int


@dataclass(frozen=True, slots=True)
class FileSearchResult:
    """一次文件搜索的当前页结果和分页元数据。"""

    items: list[FileEntry]
    total: int
    page: int
    page_size: int


_FILE_SORT_FIELDS: dict[str, FileEntrySortField] = {
    "relative_path": "relative_path",
    "name": "name",
    "size_bytes": "size_bytes",
    "modified_at": "mtime_ns",
}

OPERATION_PLAN_MAX_AGE = timedelta(minutes=15)


def _restore_operation_plan_item(item: OperationItemRecord) -> OperationPlanItem:
    """将一条持久化明细重新验证为业务契约。"""

    raw_risks = json.loads(item.risks_json)
    if not isinstance(raw_risks, list):
        raise ValueError("risks_json must contain a list")

    content_hash = None
    if (
        item.source_hash_algorithm is not None
        or item.source_sha256 is not None
    ):
        if (
            item.source_hash_algorithm is None
            or item.source_sha256 is None
        ):
            raise ValueError("source hash fields must be provided together")
        content_hash = ContentHash(
            algorithm=item.source_hash_algorithm,
            digest=item.source_sha256,
        )

    return OperationPlanItem(
        operation_type=item.operation_type,
        source_file_id=item.source_file_id,
        source_relative_path=item.source_relative_path,
        target_relative_path=item.target_relative_path,
        source_precondition=FilePrecondition(
            size_bytes=item.source_size_bytes,
            mtime_ns=item.source_mtime_ns,
            content_hash=content_hash,
        ),
        reason=OperationReason(
            kind=item.reason_kind,
            description=item.reason_description,
            match_score=item.reason_match_score,
        ),
        risks=tuple(
            OperationRisk.model_validate(raw_risk)
            for raw_risk in raw_risks
        ),
    )


def _restore_operation_plan(record: OperationPlanRecord) -> OperationPlan:
    """将一条持久化计划及其明细重新验证为业务契约。"""

    created_at = record.created_at
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = created_at.astimezone(timezone.utc)

    return OperationPlan(
        schema_version=record.schema_version,
        plan_id=UUID(record.plan_id),
        workspace_id=record.workspace_id,
        created_at=created_at,
        operations=tuple(
            _restore_operation_plan_item(item)
            for item in sorted(record.items, key=lambda current: current.sequence_no)
        ),
    )


def get_operation_plan(
    session: Session,
    plan_id: UUID | str,
    *,
    workflow_id: UUID | str | None = None,
) -> OperationPlan | None:
    """从业务库读取并严格还原完整的操作计划契约。"""

    record = get_operation_plan_by_id(session, str(plan_id))
    if record is None or (
        workflow_id is not None and record.workflow_id != str(workflow_id)
    ):
        return None

    try:
        return _restore_operation_plan(record)
    except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
        raise OperationPlanPersistenceError(
            f"操作计划 {record.plan_id} 无法安全还原"
        ) from error


def load_operation_plan_policy_snapshot(
    session: Session,
    plan_id: UUID | str,
) -> WorkspacePolicy:
    """严格读取计划创建时保存的 Workspace Policy 快照。"""

    record = get_operation_plan_by_id(session, str(plan_id))
    if record is None:
        raise OperationPlanPersistenceError(
            f"操作计划 {plan_id} 不存在"
        )
    try:
        metadata = json.loads(record.metadata_json)
        snapshot = metadata["workspace_policy"]
        if not isinstance(snapshot, dict):
            raise ValueError("workspace_policy snapshot must be an object")
        return parse_workspace_policy(
            policy_revision=snapshot["policy_revision"],
            read_enabled=snapshot["read_enabled"],
            proposal_enabled=snapshot["proposal_enabled"],
            safe_execution_enabled=snapshot["safe_execution_enabled"],
            user_denylist_json=snapshot["user_denylist_json"],
            ignore_patterns_json=snapshot["ignore_patterns_json"],
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        WorkspacePolicyPersistenceError,
    ) as error:
        raise OperationPlanPersistenceError(
            f"操作计划 {record.plan_id} 的 Workspace Policy 快照损坏"
        ) from error


def list_operation_plan_items(
    session: Session,
    plan_id: UUID | str,
) -> list[OperationPlanItem]:
    """读取并严格还原一个计划的全部操作明细。"""

    items = find_operation_plan_items(session, str(plan_id))
    try:
        return [_restore_operation_plan_item(item) for item in items]
    except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
        raise OperationPlanPersistenceError(
            f"操作计划 {plan_id} 的明细无法安全还原"
        ) from error


def list_operation_plan_history(
    session: Session,
    workflow_id: UUID | str,
) -> list[OperationPlan]:
    """读取并严格还原一个工作流关联的全部计划版本。"""

    records = find_operation_plan_history(session, str(workflow_id))
    try:
        return [_restore_operation_plan(record) for record in records]
    except (TypeError, ValueError, json.JSONDecodeError, ValidationError) as error:
        raise OperationPlanPersistenceError(
            f"工作流 {workflow_id} 的操作计划历史无法安全还原"
        ) from error


def require_approved_operation_plan(
    session: Session,
    workflow_id: UUID,
    plan: OperationPlan,
) -> ApprovalRequest:
    """只允许与持久化审批记录完全匹配的已批准计划通过。"""

    approval = get_approval_request_by_workflow_id(
        session,
        str(workflow_id),
    )
    if approval is None:
        raise OperationPlanApprovalError(
            OperationPlanApprovalErrorCode.NOT_FOUND,
            "操作计划没有对应的审批记录",
        )
    if approval.status != "APPROVED":
        raise OperationPlanApprovalError(
            OperationPlanApprovalErrorCode.NOT_APPROVED,
            "操作计划尚未获得人工批准",
        )
    if approval.plan_id != str(plan.plan_id):
        raise OperationPlanApprovalError(
            OperationPlanApprovalErrorCode.PLAN_MISMATCH,
            "已批准计划与当前操作计划不一致",
        )

    persisted_record = get_operation_plan_by_id(
        session,
        approval.plan_id,
    )
    if persisted_record is None:
        raise OperationPlanApprovalError(
            OperationPlanApprovalErrorCode.PLAN_MISMATCH,
            "已批准的业务操作计划不存在",
        )
    session.expire(persisted_record)
    if (
        persisted_record.workflow_id != str(workflow_id)
        or persisted_record.status != "APPROVED"
    ):
        raise OperationPlanApprovalError(
            OperationPlanApprovalErrorCode.PLAN_MISMATCH,
            "已批准的业务操作计划状态或归属不一致",
        )
    try:
        persisted_plan = get_operation_plan(
            session,
            approval.plan_id,
            workflow_id=workflow_id,
        )
    except OperationPlanPersistenceError as error:
        raise OperationPlanApprovalError(
            OperationPlanApprovalErrorCode.PLAN_MISMATCH,
            "已批准的业务操作计划无法还原",
        ) from error
    if persisted_plan is None or persisted_plan != plan:
        raise OperationPlanApprovalError(
            OperationPlanApprovalErrorCode.PLAN_MISMATCH,
            "业务库中的操作计划与当前操作计划不一致",
        )

    return approval


def approve_operation_plan(
    session: Session,
    workflow_id: UUID,
    expected_plan_id: UUID,
) -> ApprovalRequest:
    """批准仍处于等待状态且内容未变化的计划。"""

    return _transition_approval_request(
        session,
        workflow_id,
        expected_plan_id,
        action="approve",
        next_status="APPROVED",
        next_plan_id=expected_plan_id,
    )


def edit_operation_plan(
    session: Session,
    workflow_id: UUID,
    expected_plan_id: UUID,
    replacement_plan_id: UUID,
) -> ApprovalRequest:
    """用新计划替换当前计划，并继续等待人工审批。"""

    if replacement_plan_id == expected_plan_id:
        raise ApprovalTransitionError(
            ApprovalTransitionErrorCode.PLAN_UNCHANGED,
            "编辑后的计划必须使用新的 plan_id",
        )

    return _transition_approval_request(
        session,
        workflow_id,
        expected_plan_id,
        action="edit",
        next_status="WAITING_APPROVAL",
        next_plan_id=replacement_plan_id,
    )


def reject_operation_plan(
    session: Session,
    workflow_id: UUID,
    expected_plan_id: UUID,
) -> ApprovalRequest:
    """拒绝仍处于等待状态且内容未变化的计划。"""

    return _transition_approval_request(
        session,
        workflow_id,
        expected_plan_id,
        action="reject",
        next_status="REJECTED",
        next_plan_id=expected_plan_id,
    )


def cancel_operation_plan(
    session: Session,
    workflow_id: UUID,
    expected_plan_id: UUID,
) -> ApprovalRequest:
    """取消仍处于等待状态且内容未变化的计划。"""

    return _transition_approval_request(
        session,
        workflow_id,
        expected_plan_id,
        action="cancel",
        next_status="CANCELLED",
        next_plan_id=expected_plan_id,
    )


def _transition_approval_request(
    session: Session,
    workflow_id: UUID,
    expected_plan_id: UUID,
    *,
    action: ApprovalAction,
    next_status: ApprovalStatus,
    next_plan_id: UUID,
) -> ApprovalRequest:
    workflow_key = str(workflow_id)
    expected_plan_key = str(expected_plan_id)
    approval = get_approval_request_by_workflow_id(session, workflow_key)

    if approval is None:
        raise ApprovalTransitionError(
            ApprovalTransitionErrorCode.NOT_FOUND,
            "审批任务不存在",
        )
    if approval.plan_id != expected_plan_key:
        raise ApprovalTransitionError(
            ApprovalTransitionErrorCode.PLAN_MISMATCH,
            "待审批计划已经变化",
        )
    if _is_repeated_approval(
        approval,
        action=action,
        expected_plan_id=expected_plan_key,
    ):
        return approval
    if approval.status != "WAITING_APPROVAL":
        raise ApprovalTransitionError(
            ApprovalTransitionErrorCode.NOT_WAITING,
            "审批任务已结束，不能再次转换",
        )

    previous_status = approval.status
    previous_plan_id = approval.plan_id

    try:
        updated = compare_and_set_approval_request(
            session,
            workflow_key,
            expected_plan_key,
            next_status=next_status,
            next_plan_id=str(next_plan_id),
        )
        if not updated:
            session.rollback()
            current_approval = get_approval_request_by_workflow_id(
                session,
                workflow_key,
            )
            if (
                current_approval is not None
                and current_approval.plan_id != expected_plan_key
            ):
                raise ApprovalTransitionError(
                    ApprovalTransitionErrorCode.PLAN_MISMATCH,
                    "待审批计划已经变化",
                )
            if (
                current_approval is not None
                and _is_repeated_approval(
                    current_approval,
                    action=action,
                    expected_plan_id=expected_plan_key,
                )
            ):
                return current_approval
            raise ApprovalTransitionError(
                ApprovalTransitionErrorCode.STATE_CHANGED,
                "审批状态在提交前已经变化",
            )

        if next_plan_id == expected_plan_id:
            _sync_operation_plan_status(
                session,
                expected_plan_key,
                next_status,
            )
        else:
            _sync_operation_plan_status(
                session,
                expected_plan_key,
                "SUPERSEDED",
            )
            _sync_operation_plan_status(
                session,
                str(next_plan_id),
                "WAITING_APPROVAL",
            )
        _sync_operation_status(
            session,
            workflow_key,
            expected_status=OperationStatus.WAITING_APPROVAL,
            next_status=OperationStatus(next_status),
            plan_id=str(next_plan_id),
            approval_id=approval.id,
        )

        add_approval_audit_event(
            session,
            ApprovalAuditEvent(
                approval_request_id=approval.id,
                action=action,
                previous_status=previous_status,
                next_status=next_status,
                previous_plan_id=previous_plan_id,
                next_plan_id=str(next_plan_id),
            ),
        )
        # 审计约束失败必须发生在业务状态提交前，保证二者共同回滚。
        session.flush()

        # 原子 UPDATE 绕过了当前 ORM 实例，提交前重新加载数据库事实。
        session.expire(approval)
        session.refresh(approval)
        session.commit()
    except Exception:
        session.rollback()
        raise

    return approval


def _sync_operation_plan_status(
    session: Session,
    plan_id: str,
    next_status: OperationPlanStatus,
) -> None:
    """在存在独立业务记录时同步审批产生的计划状态。"""

    record = get_operation_plan_by_id(session, plan_id)
    if record is None:
        # 兼容迁移前仅有 ApprovalRequest 的历史记录；新建流程始终会有 Plan。
        return
    if not compare_and_set_operation_plan_status(
        session,
        plan_id,
        "WAITING_APPROVAL",
        next_status=next_status,
    ):
        raise ApprovalTransitionError(
            ApprovalTransitionErrorCode.STATE_CHANGED,
            "操作计划状态在提交前已经变化",
        )
    session.expire(record, ["status"])


def _sync_operation_status(
    session: Session,
    workflow_id: str,
    *,
    expected_status: OperationStatus,
    next_status: OperationStatus,
    plan_id: str,
    approval_id: int,
) -> None:
    """把审批业务结果同步到独立 Operation 当前状态；兼容历史缺失记录。"""

    record = get_operation_status_by_workflow_id(session, workflow_id)
    if record is None:
        return

    # CAS 更新绕过当前 ORM 实例；先刷新，避免复用旧 revision 覆盖并发事实。
    session.expire(record)
    session.refresh(record)
    if record.overall_status != expected_status.value:
        raise ApprovalTransitionError(
            ApprovalTransitionErrorCode.STATE_CHANGED,
            "Operation 状态在提交前已经变化",
        )

    if next_status == expected_status:
        updated = compare_and_set_operation_status_links(
            session,
            workflow_id,
            expected_status,
            expected_revision=record.revision,
            plan_id=plan_id,
            approval_id=approval_id,
        )
    else:
        updated = compare_and_set_operation_status(
            session,
            workflow_id,
            expected_status,
            expected_revision=record.revision,
            next_status=next_status,
            plan_id=plan_id,
            approval_id=approval_id,
        )
    if not updated:
        raise ApprovalTransitionError(
            ApprovalTransitionErrorCode.STATE_CHANGED,
            "Operation 状态在提交前已经变化",
        )
    session.expire(record)


def _is_repeated_approval(
    approval: ApprovalRequest,
    *,
    action: ApprovalAction,
    expected_plan_id: str,
) -> bool:
    """只把同一计划的重复批准或取消视为幂等成功。"""

    return (
        (
            action == "approve"
            and approval.status == "APPROVED"
        )
        or (
            action == "cancel"
            and approval.status == "CANCELLED"
        )
    ) and approval.plan_id == expected_plan_id


def load_workspace_policy(
    session: Session,
    workspace_id: int,
) -> WorkspacePolicy:
    """严格加载持久化策略；缺失或 JSON 损坏都必须失败关闭。"""

    if get_workspace_by_id(session, workspace_id) is None:
        raise WorkspaceNotFoundError(workspace_id)
    try:
        return get_workspace_policy(session, workspace_id)
    except WorkspacePolicyPersistenceError as error:
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.INVALID,
            "Workspace Policy 缺失或损坏",
        ) from error


def require_workspace_read_policy(
    session: Session,
    workspace_id: int,
) -> WorkspacePolicy:
    """加载并确认工作区允许读取；关闭时所有读取入口统一失败关闭。"""

    policy = load_workspace_policy(session, workspace_id)
    if not policy.read_enabled:
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.READ_DISABLED,
            "Workspace Policy 已禁用读取",
        )
    return policy


def require_workspace_proposal_policy(
    session: Session,
    workspace_id: int,
) -> WorkspacePolicy:
    """加载并确认工作区允许提案；提案能力不能绕过读取开关。"""

    policy = require_workspace_read_policy(session, workspace_id)
    if not policy.proposal_enabled:
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.PROPOSAL_DISABLED,
            "Workspace Policy 已禁用提案",
        )
    return policy


def update_workspace_policy(
    session: Session,
    workspace_id: int,
    policy: WorkspacePolicy,
    *,
    actor: str,
    source: str,
) -> WorkspacePolicy:
    """以 revision CAS 更新策略，并把成功变更与审计放进同一事务。"""

    if get_workspace_by_id(session, workspace_id) is None:
        raise WorkspaceNotFoundError(workspace_id)
    if (
        not isinstance(actor, str)
        or not actor
        or actor != actor.strip()
        or len(actor) > 128
        or not isinstance(source, str)
        or not source
        or source != source.strip()
        or len(source) > 128
    ):
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.INVALID,
            "Policy 变更 actor/source 不符合契约",
        )

    record = get_workspace_policy_record(session, workspace_id)
    if record is None:
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.NOT_FOUND,
            "Workspace Policy 记录不存在",
        )
    try:
        current_policy = load_workspace_policy(session, workspace_id)
    except WorkspacePolicyError:
        raise
    if policy.policy_revision != current_policy.policy_revision + 1:
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.REVISION_CONFLICT,
            "Workspace Policy revision 已经过期",
        )

    current_summary = workspace_policy_rule_summary(current_policy)
    next_summary = workspace_policy_rule_summary(policy)
    added_rules = {
        field_name: sorted(
            set(next_summary[field_name]) - set(current_summary[field_name])
        )
        for field_name in current_summary
    }
    removed_rules = {
        field_name: sorted(
            set(current_summary[field_name]) - set(next_summary[field_name])
        )
        for field_name in current_summary
    }
    updated = compare_and_set_workspace_policy(
        session,
        workspace_id,
        current_policy.policy_revision,
        policy,
    )
    if not updated:
        session.rollback()
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.REVISION_CONFLICT,
            "Workspace Policy revision 已在提交前变化",
        )

    audit_event = WorkspacePolicyAuditEvent(
        workspace_id=workspace_id,
        actor=actor,
        source=source,
        previous_revision=current_policy.policy_revision,
        next_revision=policy.policy_revision,
        added_rules_json=json.dumps(
            added_rules,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        removed_rules_json=json.dumps(
            removed_rules,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        result="succeeded",
    )
    try:
        # 审计 flush/commit 失败必须回滚前面的策略 UPDATE，不能放行新权限。
        add_workspace_policy_audit_event(session, audit_event)
        session.flush()
        session.commit()
    except Exception as error:
        session.rollback()
        raise WorkspacePolicyError(
            WorkspacePolicyErrorCode.AUDIT_WRITE_FAILED,
            "Workspace Policy 审计写入失败",
        ) from error
    return policy


def create_workspace(
    session: Session,
    name: str,
    root_path: str,
) -> Workspace:
    """创建并保存工作区；根路径重复时抛出业务冲突错误。"""

    normalized_root = validate_workspace_root(root_path)
    normalized_key = os.path.normcase(str(normalized_root))
    for existing_workspace in find_workspaces(session):
        try:
            existing_root = normalize_workspace_root(existing_workspace.root_path)
        except PathPolicyError:
            continue
        if os.path.normcase(str(existing_root)) == normalized_key:
            raise WorkspacePathConflictError

    workspace = Workspace(
        name=name,
        root_path=str(normalized_root),
    )

    add_workspace(session, workspace)

    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise WorkspacePathConflictError from error

    return workspace


def list_workspaces(
    session: Session,
    name: str | None = None,
) -> list[Workspace]:
    """查询工作区列表，可按名称筛选。"""

    return find_workspaces(session, name)


def get_workspace(
    session: Session,
    workspace_id: int,
) -> Workspace | None:
    """按 ID 查询工作区，找不到时返回 None。"""

    return get_workspace_by_id(session, workspace_id)


def get_file_detail(
    session: Session,
    workspace_id: int,
    file_id: int,
) -> FileEntry:
    """读取指定工作区内一个文件索引的详情。"""

    policy = require_workspace_read_policy(session, workspace_id)
    workspace = get_workspace_by_id(session, workspace_id)
    if workspace is None:
        raise WorkspaceNotFoundError(workspace_id)

    file_entry = get_file_entry_by_id(session, workspace_id, file_id)
    if file_entry is None:
        raise FileEntryNotFoundError(file_id)

    FileSystemAdapter(
        Path(workspace.root_path),
        workspace_policy=policy,
    ).authorized_path(Path(file_entry.relative_path))
    return file_entry


def get_authorized_file_metadata(
    session: Session,
    workspace_id: int,
    file_id: int,
) -> FileEntry:
    """读取文件索引元数据前验证工作区归属和相对路径授权。"""

    workspace = get_workspace_by_id(session, workspace_id)
    if workspace is None:
        raise WorkspaceNotFoundError(workspace_id)

    file_entry = get_file_entry_by_id(session, workspace_id, file_id)
    if file_entry is None:
        raise FileEntryNotFoundError(file_id)

    # 索引属于数据库数据，也可能因历史版本或人工修改而不可信。
    # 返回给 Agent 前重新经过 Path Policy，防止污染路径越过工作区。
    policy = require_workspace_read_policy(session, workspace_id)
    adapter = FileSystemAdapter(
        Path(workspace.root_path),
        workspace_policy=policy,
    )
    adapter.authorized_path(Path(file_entry.relative_path))
    return file_entry


def generate_operation_preview(
    session: Session,
    request: OperationPreviewRequest,
) -> OperationPreviewResponse:
    """根据当前索引和真实磁盘状态生成只读候选预览。"""

    policy = require_workspace_proposal_policy(
        session,
        request.workspace_id,
    )
    workspace = get_workspace_by_id(session, request.workspace_id)
    if workspace is None:
        raise WorkspaceNotFoundError(request.workspace_id)

    adapter = FileSystemAdapter(
        Path(workspace.root_path),
        workspace_policy=policy,
    )
    for directory in request.target_directories:
        try:
            is_directory = adapter.is_directory(Path(directory))
        except OSError as error:
            raise OperationPreviewPathUnavailableError(directory) from error
        if not is_directory:
            raise OperationPreviewPathUnavailableError(directory)

    items: list[OperationPreviewItem] = []
    for file_id in request.source_file_ids:
        file_entry = get_file_entry_by_id(
            session,
            request.workspace_id,
            file_id,
        )
        if file_entry is None:
            raise FileEntryNotFoundError(file_id)

        try:
            metadata = adapter.get_file_metadata(Path(file_entry.relative_path))
        except OSError as error:
            raise OperationPreviewPathUnavailableError(
                file_entry.relative_path
            ) from error
        if metadata is None:
            raise OperationPreviewPathUnavailableError(file_entry.relative_path)

        items.append(
            OperationPreviewItem(
                source_file_id=file_entry.id,
                source_relative_path=file_entry.relative_path,
                candidates=rank_preview_candidates(
                    file_entry.name,
                    request.target_directories,
                ),
            )
        )

    return OperationPreviewResponse(
        workspace_id=request.workspace_id,
        items=items,
    )


def validate_operation_plan(
    session: Session,
    plan: OperationPlan,
    *,
    now: datetime | None = None,
    quarantine_root: Path | None = None,
) -> None:
    """校验计划时间、源文件状态、工作区归属和目标冲突。"""

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("now must include a timezone")

    plan_age = current_time - plan.created_at
    if plan_age < timedelta(0) or plan_age > OPERATION_PLAN_MAX_AGE:
        raise OperationPlanExpiredError(plan.plan_id)

    workspace = get_workspace_by_id(session, plan.workspace_id)
    if workspace is None:
        raise WorkspaceNotFoundError(plan.workspace_id)

    policy = load_workspace_policy(session, plan.workspace_id)
    adapter = FileSystemAdapter(
        Path(workspace.root_path),
        workspace_policy=policy,
    )
    quarantine_adapter: FileSystemAdapter | None = None
    for operation in plan.operations:
        file_entry = get_file_entry_by_id(
            session,
            plan.workspace_id,
            operation.source_file_id,
        )
        if file_entry is None:
            raise FileEntryNotFoundError(operation.source_file_id)
        if file_entry.relative_path != operation.source_relative_path:
            raise OperationPlanSourceMismatchError(operation.source_file_id)

        source_path = Path(operation.source_relative_path)
        adapter.authorized_path(source_path)
        try:
            current_metadata = adapter.get_file_metadata(source_path)
        except OSError as error:
            raise OperationPlanSourceChangedError(
                operation.source_file_id
            ) from error

        expected_metadata = operation.source_precondition
        if (
            current_metadata is None
            or current_metadata.size_bytes != expected_metadata.size_bytes
            or current_metadata.mtime_ns != expected_metadata.mtime_ns
        ):
            raise OperationPlanSourceChangedError(operation.source_file_id)

        expected_hash = expected_metadata.content_hash
        if expected_hash is not None:
            try:
                current_hash = adapter.get_file_sha256(source_path)
            except OSError as error:
                raise OperationPlanSourceChangedError(
                    operation.source_file_id
                ) from error
            if current_hash != expected_hash.digest:
                raise OperationPlanSourceChangedError(operation.source_file_id)

        target_adapter = adapter
        if operation.operation_type == "quarantine":
            if quarantine_adapter is None:
                quarantine_adapter = FileSystemAdapter(
                    quarantine_root or resolve_quarantine_root()
                )
                try:
                    QuarantineManager(adapter, quarantine_adapter)
                except (PathPolicyError, QuarantineError) as error:
                    raise OperationPlanTargetUnavailableError(
                        operation.target_relative_path
                    ) from error

            expected_target = build_quarantine_relative_path(
                workspace_id=plan.workspace_id,
                plan_id=plan.plan_id,
                source_file_id=operation.source_file_id,
                file_name=source_path.name,
            ).as_posix()
            if operation.target_relative_path != expected_target:
                raise OperationPlanTargetUnavailableError(
                    operation.target_relative_path
                )
            target_adapter = quarantine_adapter

        target_path = Path(operation.target_relative_path)
        target_adapter.authorized_path(target_path)

        try:
            # 隔离目录由 QuarantineManager 在真正执行时创建；审批前只检查
            # 目标是否已被占用，避免把“待创建目录”误判为不可用。
            if operation.operation_type != "quarantine" and not adapter.is_directory(
                target_path.parent
            ):
                raise OperationPlanTargetUnavailableError(
                    operation.target_relative_path
                )
            if target_adapter.path_exists(target_path):
                raise OperationPlanTargetConflictError(
                    operation.target_relative_path
                )
        except OSError as error:
            raise OperationPlanTargetUnavailableError(
                operation.target_relative_path
            ) from error


def search_files(
    session: Session,
    workspace_id: int,
    *,
    keyword: str | None = None,
    extension: str | None = None,
    modified_from: datetime | None = None,
    modified_to: datetime | None = None,
    sort_by: str = "relative_path",
    sort_order: str = "asc",
    page: int = 1,
    page_size: int = 50,
) -> FileSearchResult:
    """在一个已授权工作区的持久化索引中搜索文件。"""

    policy = require_workspace_read_policy(session, workspace_id)
    workspace = get_workspace_by_id(session, workspace_id)
    if workspace is None:
        raise WorkspaceNotFoundError(workspace_id)
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    if sort_by not in _FILE_SORT_FIELDS:
        raise ValueError(f"unsupported file sort field: {sort_by}")

    normalized_keyword = _normalize_keyword(keyword)
    normalized_extension = _normalize_extension(extension)
    modified_from_ns = _datetime_to_epoch_ns(modified_from)
    modified_to_ns = _datetime_to_epoch_ns(modified_to)
    adapter = FileSystemAdapter(
        Path(workspace.root_path),
        workspace_policy=policy,
    )
    candidates = find_file_entries(
        session,
        workspace_id,
        sort_by=_FILE_SORT_FIELDS[sort_by],
        sort_order=cast(SortOrder, sort_order),
        offset=0,
        limit=None,
        keyword=normalized_keyword,
        extension=normalized_extension,
        modified_from_ns=modified_from_ns,
        modified_to_ns=modified_to_ns,
    )
    visible_items: list[FileEntry] = []
    for file_entry in candidates:
        try:
            adapter.authorized_path(Path(file_entry.relative_path))
        except PathPolicyError:
            continue

        visible_items.append(file_entry)

    total = len(visible_items)
    page_start = (page - 1) * page_size
    items = visible_items[page_start : page_start + page_size]

    return FileSearchResult(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


def _normalize_keyword(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        raise ValueError("keyword must not be blank")
    return normalized


def _normalize_extension(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().casefold()
    if not normalized:
        raise ValueError("extension must not be blank")
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def _datetime_to_epoch_ns(value: datetime | None) -> int | None:
    """不经过浮点时间戳，将带时区时间精确转换为纳秒。"""

    if value is None:
        return None
    if value.utcoffset() is None:
        raise ValueError("modified time must include timezone information")

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = value.astimezone(timezone.utc) - epoch
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000_000
        + delta.microseconds * 1_000
    )


def sync_file_index(
    session: Session,
    workspace_id: int,
    scanned_files: list[ScannedFile],
) -> FileIndexSyncResult:
    """将一次完整扫描结果幂等同步到指定工作区的文件索引。"""

    scanned_by_path: dict[str, ScannedFile] = {}
    for scanned_file in scanned_files:
        if scanned_file.relative_path in scanned_by_path:
            raise DuplicateScannedPathError(scanned_file.relative_path)
        scanned_by_path[scanned_file.relative_path] = scanned_file

    try:
        if get_workspace_by_id(session, workspace_id) is None:
            raise WorkspaceNotFoundError(workspace_id)

        existing_by_path = {
            file_entry.relative_path: file_entry
            for file_entry in find_file_entries(session, workspace_id)
        }
        created = 0
        updated = 0
        unchanged = 0

        for relative_path, scanned_file in scanned_by_path.items():
            file_entry = existing_by_path.get(relative_path)

            if file_entry is None:
                add_file_entry(
                    session,
                    FileEntry(
                        workspace_id=workspace_id,
                        relative_path=relative_path,
                        name=scanned_file.name,
                        extension=scanned_file.extension,
                        size_bytes=scanned_file.size_bytes,
                        mtime_ns=scanned_file.mtime_ns,
                    ),
                )
                created += 1
                continue

            current_metadata = (
                file_entry.name,
                file_entry.extension,
                file_entry.size_bytes,
                file_entry.mtime_ns,
            )
            scanned_metadata = (
                scanned_file.name,
                scanned_file.extension,
                scanned_file.size_bytes,
                scanned_file.mtime_ns,
            )

            if current_metadata == scanned_metadata:
                unchanged += 1
                continue

            file_entry.name = scanned_file.name
            file_entry.extension = scanned_file.extension
            file_entry.size_bytes = scanned_file.size_bytes
            file_entry.mtime_ns = scanned_file.mtime_ns
            updated += 1

        deleted = 0
        for relative_path, file_entry in existing_by_path.items():
            if relative_path not in scanned_by_path:
                delete_file_entry(session, file_entry)
                deleted += 1

        session.commit()
    except Exception:
        session.rollback()
        raise

    return FileIndexSyncResult(
        created=created,
        updated=updated,
        deleted=deleted,
        unchanged=unchanged,
    )


def scan_workspace(
    session: Session,
    workspace_id: int,
) -> FileIndexSyncResult:
    """安全扫描一个已授权工作区，并同步其文件索引。"""

    workspace = get_workspace_by_id(session, workspace_id)
    if workspace is None:
        session.rollback()
        raise WorkspaceNotFoundError(workspace_id)

    policy = require_workspace_read_policy(session, workspace_id)
    workspace_root = Path(workspace.root_path)
    adapter = FileSystemAdapter(
        workspace_root,
        workspace_policy=policy,
    )
    ignored_entries: list[IgnoredEntry] = []

    try:
        _require_scannable_workspace_root(adapter)
        scanned_files = scan_workspace_files(
            workspace_root,
            ignore_patterns=[],
            ignored_entries=ignored_entries,
            workspace_policy=policy,
        )
        # 扫描期间根目录失效时，空结果不能覆盖现有索引。
        _require_scannable_workspace_root(adapter)
    except WorkspaceScanUnavailableError:
        session.rollback()
        raise

    logger = logging.getLogger("FileNest")
    for ignored_entry in ignored_entries:
        logger.info(
            "工作区扫描排除条目",
            extra={
                "workspace_id": workspace_id,
                "relative_path": ignored_entry.relative_path,
                "ignored_reason": ignored_entry.ignored_reason,
            },
        )

    return sync_file_index(session, workspace_id, scanned_files)


def _require_scannable_workspace_root(
    adapter: FileSystemAdapter,
) -> None:
    """确认工作区根目录存在、是目录并通过 Path Policy。"""

    try:
        is_directory = adapter.is_directory(Path("."))
    except (OSError, PathPolicyError) as error:
        raise WorkspaceScanUnavailableError from error

    if not is_directory:
        raise WorkspaceScanUnavailableError
