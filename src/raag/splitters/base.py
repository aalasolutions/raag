"""Base splitter protocol and data structures."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SplitSection:
    """A section produced by a document splitter."""

    text: str
    heading_level: int = 0
    section_number: str | None = None
    section_title: str | None = None
    parent_path: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1


class Splitter(Protocol):
    """Protocol for document splitters."""

    def split(self, text: str, max_chars: int = 1000) -> list[SplitSection]: ...
