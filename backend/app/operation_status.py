"""Operation 统一状态契约。"""

from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping


class OperationStatus(StrEnum):
    """一个 Operation 在提议、执行和恢复过程中的统一状态。"""

    PROPOSED = "PROPOSED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXECUTING = "EXECUTING"
    PARTIAL_FAILED = "PARTIAL_FAILED"
    COMPLETED = "COMPLETED"
    UNDOING = "UNDOING"
    UNDONE = "UNDONE"
    COMPENSATED = "COMPENSATED"
    FAILED = "FAILED"


class OperationStatusTransitionErrorCode(StrEnum):
    """统一 Operation 状态转换失败时供上层稳定判断的错误码。"""

    INVALID_TRANSITION = "invalid_transition"


class OperationStatusTransitionError(ValueError):
    """拒绝不符合统一 Operation 状态图的转换。"""

    def __init__(
        self,
        current_status: OperationStatus,
        next_status: OperationStatus,
    ) -> None:
        super().__init__(
            f"状态 {current_status.value} 不允许转换为 {next_status.value}"
        )
        self.code = OperationStatusTransitionErrorCode.INVALID_TRANSITION
        self.current_status = current_status
        self.next_status = next_status


OPERATION_STATUS_DEFINITIONS: Final[Mapping[OperationStatus, str]] = (
    MappingProxyType(
        {
            OperationStatus.PROPOSED: "已生成操作提议，但尚未进入人工审批状态。",
            OperationStatus.WAITING_APPROVAL: "操作计划等待人工审批。",
            OperationStatus.APPROVED: "操作计划已获批准，尚未进入文件副作用边界。",
            OperationStatus.REJECTED: "人工拒绝了操作计划。",
            OperationStatus.CANCELLED: "操作在未完成前被明确取消。",
            OperationStatus.EXECUTING: "操作正在执行文件变更。",
            OperationStatus.PARTIAL_FAILED: "部分操作失败，仍保留重试或补偿证据。",
            OperationStatus.COMPLETED: "全部计划项已执行完成，可继续发起显式撤销。",
            OperationStatus.UNDOING: "正在撤销或补偿已经产生的文件变更。",
            OperationStatus.UNDONE: "显式撤销已完成。",
            OperationStatus.COMPENSATED: "部分失败后的补偿已完成。",
            OperationStatus.FAILED: "当前执行尝试失败，但可依据持久化证据重试。",
        }
    )
)


# FAILED -> EXECUTING 保留现有执行历史的重试语义；进入文件副作用边界后，
# 不允许直接转为 CANCELLED，避免尚未对账的磁盘结果被误标为已取消。
OPERATION_STATUS_TRANSITIONS: Final[
    Mapping[OperationStatus, frozenset[OperationStatus]]
] = MappingProxyType(
    {
        OperationStatus.PROPOSED: frozenset(
            {OperationStatus.WAITING_APPROVAL, OperationStatus.CANCELLED}
        ),
        OperationStatus.WAITING_APPROVAL: frozenset(
            {
                OperationStatus.APPROVED,
                OperationStatus.REJECTED,
                OperationStatus.CANCELLED,
            }
        ),
        OperationStatus.APPROVED: frozenset(
            {OperationStatus.EXECUTING, OperationStatus.CANCELLED}
        ),
        OperationStatus.REJECTED: frozenset(),
        OperationStatus.CANCELLED: frozenset(),
        OperationStatus.EXECUTING: frozenset(
            {
                OperationStatus.PARTIAL_FAILED,
                OperationStatus.COMPLETED,
                OperationStatus.FAILED,
            }
        ),
        OperationStatus.PARTIAL_FAILED: frozenset(
            {
                OperationStatus.EXECUTING,
                OperationStatus.UNDOING,
                OperationStatus.FAILED,
            }
        ),
        OperationStatus.COMPLETED: frozenset({OperationStatus.UNDOING}),
        OperationStatus.UNDOING: frozenset(
            {
                OperationStatus.UNDONE,
                OperationStatus.COMPENSATED,
                OperationStatus.FAILED,
            }
        ),
        OperationStatus.UNDONE: frozenset(),
        OperationStatus.COMPENSATED: frozenset(),
        OperationStatus.FAILED: frozenset({OperationStatus.EXECUTING}),
    }
)


TERMINAL_OPERATION_STATUSES: Final[frozenset[OperationStatus]] = frozenset(
    {
        OperationStatus.REJECTED,
        OperationStatus.CANCELLED,
        OperationStatus.UNDONE,
        OperationStatus.COMPENSATED,
    }
)


def describe_operation_status(
    status: OperationStatus | str,
) -> str:
    """返回一个已知统一状态的稳定业务定义。"""

    normalized_status = _coerce_status(status)
    return OPERATION_STATUS_DEFINITIONS[normalized_status]


def is_terminal_operation_status(
    status: OperationStatus | str,
) -> bool:
    """判断统一状态是否不再接受任何业务转换。"""

    normalized_status = _coerce_status(status)
    return normalized_status in TERMINAL_OPERATION_STATUSES


def transition_operation_status(
    current_status: OperationStatus | str,
    next_status: OperationStatus | str,
) -> OperationStatus:
    """验证一次纯状态转换，不执行数据库或文件系统操作。"""

    current = _coerce_status(current_status)
    next_value = _coerce_status(next_status)
    if next_value not in OPERATION_STATUS_TRANSITIONS[current]:
        raise OperationStatusTransitionError(current, next_value)
    return next_value


def _coerce_status(status: OperationStatus | str) -> OperationStatus:
    try:
        return OperationStatus(status)
    except ValueError as error:
        raise ValueError(f"未知 Operation 状态: {status}") from error


_WORKFLOW_STATUS_TO_OPERATION_STATUS: Final[Mapping[str, OperationStatus]] = {
    # In the organization workflow, a ready workflow has already passed approval.
    "ready": OperationStatus.APPROVED,
    "waiting": OperationStatus.WAITING_APPROVAL,
    "completed": OperationStatus.COMPLETED,
    "failed": OperationStatus.FAILED,
    "cancelled": OperationStatus.CANCELLED,
}

_APPROVAL_STATUS_TO_OPERATION_STATUS: Final[Mapping[str, OperationStatus]] = {
    "WAITING_APPROVAL": OperationStatus.WAITING_APPROVAL,
    "APPROVED": OperationStatus.APPROVED,
    "REJECTED": OperationStatus.REJECTED,
    "CANCELLED": OperationStatus.CANCELLED,
}

_AGENT_RUN_STATUS_TO_OPERATION_STATUS: Final[Mapping[str, OperationStatus]] = {
    "completed": OperationStatus.COMPLETED,
    "cancelled": OperationStatus.CANCELLED,
    "failed": OperationStatus.FAILED,
    "timed_out": OperationStatus.FAILED,
    "max_steps_reached": OperationStatus.FAILED,
}


def map_workflow_status_to_operation_status(
    status: str,
    *,
    error_code: str | None = None,
) -> OperationStatus:
    """将 Workflow 的持久化状态映射为统一 Operation 状态。"""

    if status == "failed" and error_code == "human_rejected":
        return OperationStatus.REJECTED
    try:
        return _WORKFLOW_STATUS_TO_OPERATION_STATUS[status]
    except KeyError as exc:
        raise ValueError(f"未知 Workflow 状态: {status}") from exc


def map_approval_status_to_operation_status(status: str) -> OperationStatus:
    """将 Approval 的业务结果映射为统一 Operation 状态。"""

    try:
        return _APPROVAL_STATUS_TO_OPERATION_STATUS[status]
    except KeyError as exc:
        raise ValueError(f"未知 Approval 状态: {status}") from exc


def map_agent_run_status_to_operation_status(status: str) -> OperationStatus:
    """将 Agent run 生命周期状态映射为统一 Operation 状态。"""

    try:
        return _AGENT_RUN_STATUS_TO_OPERATION_STATUS[status]
    except KeyError as exc:
        raise ValueError(f"未知 Agent run 状态: {status}") from exc
