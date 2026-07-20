from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.model_settings import ModelSettings


MODEL_ENVIRONMENT_VARIABLES = (
    "FILENEST_MODEL_PROVIDER",
    "FILENEST_MODEL_NAME",
    "FILENEST_MODEL_API_KEY",
)


def _clear_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in MODEL_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_model_settings_load_required_values_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("FILENEST_MODEL_PROVIDER", "example-provider")
    monkeypatch.setenv("FILENEST_MODEL_NAME", "example-model")
    monkeypatch.setenv("FILENEST_MODEL_API_KEY", "secret-for-test")

    settings = ModelSettings()

    assert settings.provider == "example-provider"
    assert settings.name == "example-model"
    assert settings.api_key.get_secret_value() == "secret-for-test"


def test_model_settings_require_an_explicit_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("FILENEST_MODEL_PROVIDER", "example-provider")
    monkeypatch.setenv("FILENEST_MODEL_NAME", "example-model")

    with pytest.raises(ValidationError) as error_info:
        ModelSettings()

    assert "api_key" in str(error_info.value)


def test_model_settings_mask_the_api_key_in_common_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_model_environment(monkeypatch)
    exposed_secret = "secret-that-must-stay-hidden"
    monkeypatch.setenv("FILENEST_MODEL_PROVIDER", "example-provider")
    monkeypatch.setenv("FILENEST_MODEL_NAME", "example-model")
    monkeypatch.setenv("FILENEST_MODEL_API_KEY", exposed_secret)

    settings = ModelSettings()

    assert exposed_secret not in repr(settings)
    assert exposed_secret not in settings.model_dump_json()
    assert str(settings.api_key) == "**********"


def test_invalid_api_key_is_masked_in_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_model_environment(monkeypatch)
    exposed_secret = "secret-with-surrounding-space"
    monkeypatch.setenv("FILENEST_MODEL_PROVIDER", "example-provider")
    monkeypatch.setenv("FILENEST_MODEL_NAME", "example-model")
    monkeypatch.setenv("FILENEST_MODEL_API_KEY", f" {exposed_secret} ")

    with pytest.raises(ValidationError) as error_info:
        ModelSettings()

    assert exposed_secret not in str(error_info.value)


def test_gitignore_protects_local_environment_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    patterns = set(
        (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    )

    assert ".env" in patterns
    assert ".env.*" in patterns
    assert "!.env.example" in patterns
