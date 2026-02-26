"""Search result models for RAAG."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    """Represents a single search result from a vector query.

    Contains the matched section along with its relevance score.
    """

    section_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict
