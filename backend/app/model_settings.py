"""真实模型客户端使用的环境变量配置。"""

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """从环境读取供应商、模型名称和受保护的 API 密钥。"""

    model_config = SettingsConfigDict(
        env_prefix="FILENEST_MODEL_",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    provider: str
    name: str
    api_key: SecretStr

    @field_validator("provider", "name")
    @classmethod
    def reject_invalid_identifier(cls, value: str) -> str:
        """拒绝空值和首尾空白，避免选择错误的供应商或模型。"""

        if not value or value != value.strip():
            raise ValueError("must be non-empty without surrounding whitespace")
        return value

    @field_validator("api_key")
    @classmethod
    def reject_invalid_api_key(cls, value: SecretStr) -> SecretStr:
        """在保持密钥遮罩的前提下拒绝空值和复制产生的空白。"""

        secret = value.get_secret_value()
        if not secret or secret != secret.strip():
            raise ValueError("must be non-empty without surrounding whitespace")
        return value
