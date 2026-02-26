"""Configuration dataclass for Raag."""

from dataclasses import dataclass


@dataclass
class RaagConfig:
    """Configuration settings for the RAAG system.

    Attributes:
        embed_batch_size: Number of texts to embed in a single batch
        default_k: Default number of results to return from search
    """

    embed_batch_size: int = 100
    default_k: int = 10
    max_section_chars: int = 1000
