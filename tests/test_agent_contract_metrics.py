from decimal import Decimal
from pathlib import Path

import pytest

from backend.app.agent_contract_dataset import (
    AgentContractToolCall,
    load_agent_contract_dataset,
)
from backend.app.agent_contract_metrics import (
    AgentContractMetricsError,
    AgentContractRunObservation,
    calculate_agent_contract_metrics,
)


DATASET_PATH = (
    Path(__file__).parents[1]
    / "backend"
    / "evaluation"
    / "agent_contract_v1.json"
)


def _call(name: str, arguments: dict[str, object]) -> AgentContractToolCall:
    return AgentContractToolCall(name=name, arguments=arguments)


def test_agent_contract_metrics_calculate_required_rates_and_latency() -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)
    observations = (
        AgentContractRunObservation(
            case_id="tool-select-filename-search",
            actual_tool_calls=(
                _call(
                    "search_files",
                    {"workspace_id": 1, "keyword": "2026-budget"},
                ),
            ),
            actual_result="completed",
            valid_parameter_calls=1,
            parameter_call_total=1,
            unauthorized_disk_changes=0,
            actual_source_paths=("finance/2026-budget.csv",),
            end_to_end_success=True,
            latency_ms=10,
        ),
        AgentContractRunObservation(
            case_id="proposal-valid-move",
            actual_tool_calls=(
                _call(
                    "search_files",
                    {"workspace_id": 1, "keyword": "2026-q1-summary"},
                ),
                _call(
                    "propose_move",
                    {
                        "workspace_id": 1,
                        "source_file_id": 1,
                        "destination": "reports/archive",
                    },
                ),
            ),
            actual_result="proposal_waiting_approval",
            valid_parameter_calls=2,
            parameter_call_total=2,
            proposal_valid=True,
            approval_intercepted=True,
            unauthorized_disk_changes=0,
            actual_source_paths=("reports/2026-q1-summary.txt",),
            end_to_end_success=True,
            latency_ms=20,
        ),
        AgentContractRunObservation(
            case_id="security-cross-workspace",
            actual_tool_calls=(
                _call(
                    "search_files",
                    {"workspace_id": 2, "keyword": "project"},
                ),
            ),
            actual_result="rejected_invalid_arguments",
            valid_parameter_calls=0,
            parameter_call_total=1,
            unauthorized_disk_changes=0,
            end_to_end_success=True,
            latency_ms=30,
        ),
        AgentContractRunObservation(
            case_id="rag-no-evidence-refusal",
            actual_tool_calls=(
                _call(
                    "knowledge_search",
                    {
                        "workspace_id": 1,
                        "query": "roadmap-2030",
                        "top_k": 5,
                    },
                ),
            ),
            actual_result="no_evidence_refusal",
            valid_parameter_calls=1,
            parameter_call_total=1,
            unauthorized_disk_changes=0,
            no_evidence_refusal=True,
            end_to_end_success=True,
            latency_ms=40,
        ),
    )

    metrics = calculate_agent_contract_metrics(
        dataset,
        observations,
        model_source="scripted_fake",
    )

    assert metrics.case_count == 4
    assert (metrics.tool_selection_correct, metrics.tool_selection_total) == (5, 5)
    assert metrics.tool_selection_accuracy == 1
    assert (metrics.argument_matches, metrics.argument_total) == (5, 5)
    assert metrics.argument_accuracy == 1
    assert (metrics.valid_parameter_calls, metrics.parameter_call_total) == (4, 5)
    assert metrics.argument_validity_rate == 0.8
    assert metrics.proposal_validity_rate == 1
    assert metrics.approval_interception_rate == 1
    assert metrics.unauthorized_disk_changes == 0
    assert metrics.unauthorized_disk_changes_gate_passed is True
    assert metrics.citation_precision == 1
    assert metrics.citation_coverage == 1
    assert metrics.no_evidence_refusal_rate == 1
    assert metrics.end_to_end_success_rate == 1
    assert metrics.latency_p50_ms == 25
    assert metrics.latency_p95_ms == 38.5
    assert metrics.estimated_cost_usd is None


def test_agent_contract_metrics_keep_real_usage_and_fail_closed_on_disk_change() -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)
    observation = AgentContractRunObservation(
        case_id="tool-select-filename-search",
        actual_result="completed",
        valid_parameter_calls=0,
        parameter_call_total=0,
        unauthorized_disk_changes=2,
        end_to_end_success=False,
        latency_ms=12.5,
        input_tokens=100,
        output_tokens=20,
        estimated_cost_usd=Decimal("0.004"),
    )

    metrics = calculate_agent_contract_metrics(
        dataset,
        (observation,),
        model_source="real_model",
    )

    assert metrics.unauthorized_disk_changes == 2
    assert metrics.unauthorized_disk_changes_gate_passed is False
    assert metrics.usage_case_count == 1
    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 20
    assert metrics.estimated_cost_usd == Decimal("0.004")
    assert metrics.end_to_end_success_rate == 0

    with pytest.raises(
        AgentContractMetricsError,
        match="scripted_fake",
    ):
        calculate_agent_contract_metrics(
            dataset,
            (observation,),
            model_source="scripted_fake",
        )


def test_agent_contract_metrics_reject_duplicate_observations() -> None:
    dataset = load_agent_contract_dataset(DATASET_PATH)
    observation = AgentContractRunObservation(
        case_id="tool-select-filename-search",
        actual_result="completed",
        valid_parameter_calls=0,
        parameter_call_total=0,
        unauthorized_disk_changes=0,
        end_to_end_success=True,
        latency_ms=1,
    )

    with pytest.raises(
        AgentContractMetricsError,
        match="case ids must be unique",
    ):
        calculate_agent_contract_metrics(
            dataset,
            (observation, observation),
            model_source="scripted_fake",
        )
