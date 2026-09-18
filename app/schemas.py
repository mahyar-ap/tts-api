"""Pydantic request and response schemas."""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.config import get_settings

TextInput = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=get_settings().max_text_length,
    ),
]


class TTSRequest(BaseModel):
    """Synthesis text bounded by the configured 10,000-character ceiling."""

    model_config = ConfigDict(extra="forbid")

    text: TextInput


class JobStatus(str, Enum):
    """Possible states for an asynchronous synthesis job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobCreated(BaseModel):
    """Response returned after a job is accepted."""

    job_id: str
    status: JobStatus
    status_url: str
    audio_url: str


class JobResponse(BaseModel):
    """Public job status representation."""

    job_id: str
    model_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    status_url: str
    audio_url: str


class ModelInfo(BaseModel):
    """Description of one exposed model."""

    id: str
    max_text_length: int


class ModelsResponse(BaseModel):
    """List of available model adapters."""

    models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Service health and queue state."""

    status: str
    queue_size: int
    queue_capacity: int
    worker_count: int
    loaded_models: list[str]


class ModelLifecycleResponse(BaseModel):
    """Result of an explicit model load or unload request."""

    model_id: str
    loaded_models: list[str]


class ErrorResponse(BaseModel):
    """Standard FastAPI error shape used in endpoint documentation."""

    detail: str
