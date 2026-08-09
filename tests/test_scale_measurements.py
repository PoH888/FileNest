from pathlib import Path

import pytest

from backend.app.scale_measurements import (
    MeasurementOutputError,
    measure_scale,
    write_measurement_report,
)


def test_small_scale_measurement_records_all_three_actions() -> None:
    measurement = measure_scale("small", repeats=1)

    assert measurement.file_count == 100
    assert measurement.document_file_count == 80
    assert measurement.search_total == 100
    assert len(measurement.scan.samples_ms) == 1
    assert len(measurement.search.samples_ms) == 1
    assert len(measurement.agent.samples_ms) == 1
    assert measurement.agent_statuses == ("completed",)
    assert measurement.agent_model_turns == (2,)
    assert measurement.agent_tool_calls == (1,)
    assert len(measurement.fixture_content_sha256) == 64


def test_measurement_report_refuses_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "measurement.json"
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(
        MeasurementOutputError,
        match="must not already exist",
    ):
        write_measurement_report({"schema_version": "1.0"}, output_path)

    assert output_path.read_text(encoding="utf-8") == "keep me"
