"""Section model for RAAG."""

from datetime import datetime, timezone
from uuid import uuid4
from pydantic import BaseModel, Field


class Section(BaseModel):
    """Represents a section of a document.

    Sections are created by splitting documents for embedding.
    Each section has a reference to its parent document.
    """

    tenant_id: str
    section_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {
        "arbitrary_types_allowed": True,
    }
