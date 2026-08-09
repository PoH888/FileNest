from pathlib import Path

import pytest

from backend.app.scale_document_measurements import (
    MeasurementOutputError,
    measure_document_scale,
    write_document_measurement_report,
)


def test_small_document_measurement_records_index_memory_and_database() -> None:
    measurement = measure_document_scale("small", repeats=1)

    assert measurement.file_count == 100
    assert measurement.source_document_file_count == 80
    assert measurement.indexed_document_count == 80
    assert measurement.indexed_chunk_count > measurement.indexed_document_count
    assert len(measurement.index.samples_ms) == 1
    assert measurement.peak_python_allocations.minimum > 0
    assert measurement.database_bytes.minimum > 0


def test_document_measurement_report_refuses_existing_output(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "measurement.json"
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(
        MeasurementOutputError,
        match="must not already exist",
    ):
        write_document_measurement_report(
            {"schema_version": "1.0"},
            output_path,
        )

    assert output_path.read_text(encoding="utf-8") == "keep me"
