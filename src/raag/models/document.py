"""Document model for RAAG."""

from datetime import datetime, timezone
from uuid import uuid4
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in the RAAG system.

    A document is the top-level container for content.
    It contains metadata and can be split into sections for embedding.
    """

    tenant_id: str
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "arbitrary_types_allowed": True,
    }
