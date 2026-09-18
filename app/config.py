"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API and model adapters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
    )

    hf_token: str | None = None
    max_text_length: int = Field(default=10000, ge=1, le=10000)
    max_queue_size: int = Field(default=100, ge=1)
    worker_count: int = Field(default=1, ge=1)
    output_dir: Path = Path("outputs")
    device: str = "cuda"
    job_ttl_seconds: int = Field(default=86400, ge=0)
    job_cleanup_interval_seconds: int = Field(default=300, ge=1)

    api_key_enabled: bool = False
    api_key: str | None = None
    api_key_header: str = Field(default="X-API-Key", min_length=1)

    model_cache_dir: Path = Path("model_cache")
    model_unload_policy: Literal["none", "single-heavy", "single-model"] = (
        "single-model"
    )
    retry_after_seconds: int = Field(default=5, ge=1)
    chatterbox_chunk_size: int = Field(default=300, ge=1, le=300)
    chatterbox_startup_timeout_seconds: int = Field(default=300, ge=1)
    chatterbox_inference_timeout_seconds: int = Field(default=1800, ge=1)
    manatts_repo_dir: Path = Path("third_party/Persian-MultiSpeaker-Tacotron2")
    manatts_python: str = "python3.11"
    manatts_package_dir: Path = Path(".manatts-packages")
    manatts_startup_timeout_seconds: int = Field(default=300, ge=1)
    manatts_inference_timeout_seconds: int = Field(default=300, ge=1)
    espeak_ng_data_dir: Path | None = None
    sherpa_num_threads: int = Field(default=2, ge=1)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
