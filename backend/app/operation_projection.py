"""统一 Operation 查询投影契约。"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .operation_status import OperationStatus


class OperationProjection(BaseModel):
    """把 Operation 的关联标识和当前总体状态收敛成只读快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow_id: UUID
    plan_id: UUID
    approval_id: int | None = Field(default=None, ge=1)
    execution_id: int | None = Field(default=None, ge=1)
    overall_status: OperationStatus
    revision: int = Field(default=0, ge=0)
