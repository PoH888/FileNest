"""Agent 服务启动时发现未完成运行的只读扫描。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from .agent_api import AGENT_RUN_RESUMABLE_STATUSES
from .agent_observability import SqlAlchemyAgentRunRecorder
from .models import AgentRun


AGENT_RUN_STARTUP_SCAN_STATUSES = frozenset(
    {
        "running",
        "waiting_approval",
        *AGENT_RUN_RESUMABLE_STATUSES,
    }
)


def scan_unfinished_agent_runs(session: Session) -> tuple[AgentRun, ...]:
    """按稳定主键顺序返回启动时需要关注的未完成 AgentRun。"""

    statement = (
        select(AgentRun)
        .where(AgentRun.status.in_(AGENT_RUN_STARTUP_SCAN_STATUSES))
        .order_by(AgentRun.id)
    )
    return tuple(session.scalars(statement))


def recover_unfinished_agent_runs(session: Session) -> tuple[int, ...]:
    """识别启动时的未完成运行，并将陈旧 running 标记为可恢复失败。"""

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
