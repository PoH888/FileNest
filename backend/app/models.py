from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)

from sqlalchemy.orm import Mapped, mapped_column
# Mapped 用来标记：这个类属性不是普通属性，而是需要映射到数据库字段的 ORM 属性。

from .database import Base

class Workspace(Base): # 继承 FileNest 的 ORM 基类
    """告诉 SQLAlchemy：
    FileNest 有一种叫 Workspace 的数据库对象。"""
    __tablename__ = "workspaces" # 保存在 SQLite 的 workspaces 表

    id: Mapped[int] = mapped_column(primary_key=True)
#       ORM 映射属性；Python 中对应整数 int
#                     该字段是主键

    name: Mapped[str] = mapped_column(String, nullable=False)
#                                     数据库不允许这个字段保存 NULL

    root_path: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True, # 给 root_path 添加唯一约束，SQLite 会拒绝第二条记录
    )


class FileEntry(Base):
    """工作区内一个文件的持久化索引记录。"""

    __tablename__ = "file_entries"
    __table_args__ = (
        # 同一相对路径只能代表工作区内的一个文件，但不同工作区可以使用相同路径。
        UniqueConstraint(
            "workspace_id",
            "relative_path",
            name="uq_file_entries_workspace_relative_path",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        nullable=False,
    )
    relative_path: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    extension: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)


class AgentRun(Base):
    """一次 Agent Loop 运行的持久化生命周期。"""

    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'max_steps_reached', "
            "'timed_out', 'cancelled', 'failed')",
            name="ck_agent_runs_status",
        ),
        CheckConstraint(
            "model_turns >= 0",
            name="ck_agent_runs_model_turns_non_negative",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="running",
        server_default="running",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    model_turns: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        server_default="0",
    )
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)


class AgentToolCall(Base):
    """一次 Agent Run 中可观察但不含原始参数和结果的工具调用。"""

    __tablename__ = "agent_tool_calls"
    __table_args__ = (
        UniqueConstraint(
            "agent_run_id",
            "sequence_no",
            name="uq_agent_tool_calls_run_sequence",
        ),
        UniqueConstraint(
            "agent_run_id",
            "model_call_id",
            name="uq_agent_tool_calls_run_model_call_id",
        ),
        CheckConstraint(
            "sequence_no >= 1",
            name="ck_agent_tool_calls_sequence_positive",
        ),
        CheckConstraint(
            "status IN ('requested', 'succeeded', 'rejected', 'failed')",
            name="ck_agent_tool_calls_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_run_id: Mapped[int] = mapped_column(
        ForeignKey("agent_runs.id"),
        nullable=False,
    )
    sequence_no: Mapped[int] = mapped_column(nullable=False)
    model_call_id: Mapped[str] = mapped_column(String, nullable=False)
    tool_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="requested",
        server_default="requested",
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)


class ApprovalRequest(Base):
    """一个等待人工决定的文件操作计划。"""

    __tablename__ = "approval_requests"
    __table_args__ = (
        UniqueConstraint(
            "workflow_id",
            name="uq_approval_requests_workflow_id",
        ),
        CheckConstraint(
            "status IN ('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
            name="ck_approval_requests_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(36), nullable=False)
    plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="WAITING_APPROVAL",
        server_default="WAITING_APPROVAL",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )


class ApprovalAuditEvent(Base):
    """一次只追加、不保存原始用户文本的审批转换记录。"""

    __tablename__ = "approval_audit_events"
    __table_args__ = (
        CheckConstraint(
            "action IN ('approve', 'edit', 'reject')",
            name="ck_approval_audit_events_action",
        ),
        CheckConstraint(
            "previous_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
            name="ck_approval_audit_events_previous_status",
        ),
        CheckConstraint(
            "next_status IN "
            "('WAITING_APPROVAL', 'APPROVED', 'REJECTED')",
            name="ck_approval_audit_events_next_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    approval_request_id: Mapped[int] = mapped_column(
        ForeignKey("approval_requests.id"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(String, nullable=False)
    previous_status: Mapped[str] = mapped_column(String, nullable=False)
    next_status: Mapped[str] = mapped_column(String, nullable=False)
    previous_plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    next_plan_id: Mapped[str] = mapped_column(String(36), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
    )
