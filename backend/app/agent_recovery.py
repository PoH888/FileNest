"""Agent 服务启动时发现未完成运行并验证可恢复边界。"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .agent_observability import (
    AgentObservabilityError,
    SqlAlchemyAgentRunRecorder,
    validate_persisted_agent_message_payload,
    validate_persisted_agent_step_summary,
)
from .models import (
    AgentMessage,
    AgentModelRun,
    AgentRun,
    AgentSession,
    AgentStep,
    agent_run_sessions,
)


AGENT_RUN_RESUMABLE_STATUSES = frozenset(
    {"cancelled", "failed", "timed_out", "max_steps_reached"}
)


AGENT_RUN_STARTUP_SCAN_STATUSES = frozenset(
    {
        "running",
        "waiting_approval",
        *AGENT_RUN_RESUMABLE_STATUSES,
    }
)


@dataclass(frozen=True, slots=True)
class AgentRecoverySnapshot:
    """启动扫描使用的脱敏恢复事实，不携带原始 prompt 或工具载荷。"""

    agent_run_id: int
    status: str
    workspace_id: int | None
    agent_session_id: int | None
    model_turns: int
    context_message_count: int
    last_completed_step_index: int | None
    incomplete_step_index: int | None
    model_run_ids: tuple[int, ...]
    can_resume: bool
    recovery_code: str | None


def scan_unfinished_agent_runs(session: Session) -> tuple[AgentRun, ...]:
    """按稳定主键顺序返回启动时需要关注的未完成 AgentRun。"""

    statement = (
        select(AgentRun)
        .where(AgentRun.status.in_(AGENT_RUN_STARTUP_SCAN_STATUSES))
        .order_by(AgentRun.id)
    )
    return tuple(session.scalars(statement))


def inspect_agent_run_recovery(
    session: Session,
    agent_run_id: int,
) -> AgentRecoverySnapshot:
    """读取 Session、Step、Message、ModelRun 与 checkpoint 的安全摘要。"""

    agent_run = session.get(AgentRun, agent_run_id)
    if agent_run is None:
        raise AgentObservabilityError("Agent 运行记录不存在")

    session_ids = tuple(
        session.scalars(
            select(AgentSession.id)
            .join(
                agent_run_sessions,
                agent_run_sessions.c.agent_session_id == AgentSession.id,
            )
            .where(agent_run_sessions.c.agent_run_id == agent_run_id)
            .order_by(AgentSession.id)
        )
    )
    session_id = session_ids[0] if len(session_ids) == 1 else None
    recovery_code: str | None = None
    context_message_count = 0
    try:
        messages, context_model_turns = SqlAlchemyAgentRunRecorder(
            session
        ).load_context(agent_run_id)
        context_message_count = len(messages)
        if context_model_turns != agent_run.model_turns:
            recovery_code = "context_turn_count_mismatch"
    except AgentObservabilityError:
        context_model_turns = agent_run.model_turns
        recovery_code = "agent_context_invalid"

    if len(session_ids) == 0:
        recovery_code = recovery_code or "agent_session_missing"
    elif len(session_ids) > 1:
        recovery_code = recovery_code or "agent_session_ambiguous"

    steps: tuple[AgentStep, ...] = ()
    model_runs: tuple[AgentModelRun, ...] = ()
    if session_id is not None:
        agent_session = session.get(AgentSession, session_id)
        if agent_session is None:
            recovery_code = recovery_code or "agent_session_missing"
        elif agent_session.workspace_id != agent_run.workspace_id:
            recovery_code = recovery_code or "agent_workspace_mismatch"
        steps = tuple(
            session.scalars(
                select(AgentStep)
                .where(AgentStep.agent_session_id == session_id)
                .order_by(AgentStep.step_index)
            )
        )
        step_ids = tuple(step.id for step in steps)
        if step_ids:
            model_runs = tuple(
                session.scalars(
                    select(AgentModelRun)
                    .where(AgentModelRun.agent_step_id.in_(step_ids))
                    .order_by(AgentModelRun.id)
                )
            )

    step_indexes = tuple(step.step_index for step in steps)
    if agent_run.model_turns > 0:
        expected_indexes = set(range(agent_run.model_turns))
        if not expected_indexes.issubset(step_indexes):
            recovery_code = recovery_code or "step_missing"

    for step in steps:
        try:
            validate_persisted_agent_step_summary(
                step.input,
                step.output_summary,
            )
        except AgentObservabilityError:
            recovery_code = recovery_code or "step_summary_invalid"
        messages_for_step = tuple(
            session.scalars(
                select(AgentMessage)
                .where(AgentMessage.agent_step_id == step.id)
                .order_by(AgentMessage.sequence_no)
            )
        )
        if tuple(message.sequence_no for message in messages_for_step) != tuple(
            range(len(messages_for_step))
        ):
            recovery_code = recovery_code or "message_sequence_invalid"
        for message in messages_for_step:
            try:
                validate_persisted_agent_message_payload(
                    message.message_type,
                    message.payload_json,
                )
            except AgentObservabilityError:
                recovery_code = recovery_code or "message_payload_invalid"

    if agent_run.model_turns > 0 and not model_runs:
        recovery_code = recovery_code or "model_run_missing"

    completed_indexes = tuple(
        step.step_index for step in steps if step.status == "completed"
    )
    incomplete_indexes = tuple(
        step.step_index
        for step in steps
        if step.status in {"pending", "running"}
    )
    return AgentRecoverySnapshot(
        agent_run_id=agent_run.id,
        status=agent_run.status,
        workspace_id=agent_run.workspace_id,
        agent_session_id=session_id,
        model_turns=agent_run.model_turns,
        context_message_count=context_message_count,
        last_completed_step_index=(
            max(completed_indexes) if completed_indexes else None
        ),
        incomplete_step_index=(
            min(incomplete_indexes) if incomplete_indexes else None
        ),
        model_run_ids=tuple(model_run.id for model_run in model_runs),
        can_resume=recovery_code is None,
        recovery_code=recovery_code,
    )


def scan_agent_recovery_snapshots(
    session: Session,
) -> tuple[AgentRecoverySnapshot, ...]:
    """按启动扫描顺序返回所有未完成运行的恢复判定。"""

    return tuple(
        inspect_agent_run_recovery(session, agent_run.id)
        for agent_run in scan_unfinished_agent_runs(session)
    )


def recover_unfinished_agent_runs(session: Session) -> tuple[int, ...]:
    """识别未完成运行，并将仍为 running 的旧 worker 标为可恢复失败。"""

    agent_runs = scan_unfinished_agent_runs(session)
    recorder = SqlAlchemyAgentRunRecorder(session)
    for agent_run in agent_runs:
        if agent_run.status != "running":
            continue
        recorder.finish_run(
            agent_run_id=agent_run.id,
            status="failed",
            model_turns=agent_run.model_turns,
            error_code="worker_interrupted",
        )
    return tuple(agent_run.id for agent_run in agent_runs)
