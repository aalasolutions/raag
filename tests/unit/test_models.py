"""Unit tests for RAAG models."""

import pytest
from datetime import datetime
from uuid import uuid4
from pydantic import ValidationError

from raag.models import Document, Section, SearchResult
from raag.config import RaagConfig


class TestDocument:
    """Tests for Document model."""

    def test_document_creation_with_auto_generated_doc_id(self):
        """Test Document creation with auto-generated doc_id."""
        doc = Document(
            tenant_id="tenant_123",
            title="Test Document"
        )

        assert doc.tenant_id == "tenant_123"
        assert doc.title == "Test Document"
        assert doc.doc_id is not None
        assert len(doc.doc_id) > 0
        assert isinstance(doc.created_at, datetime)
        assert doc.metadata == {}

    def test_document_creation_with_custom_doc_id(self):
        """Test Document creation with custom doc_id."""
        custom_id = "custom_doc_id"
        doc = Document(
            tenant_id="tenant_123",
            doc_id=custom_id,
            title="Test Document"
        )

        assert doc.doc_id == custom_id

    def test_document_with_metadata(self):
        """Test Document creation with metadata."""
        metadata = {"author": "AAMIR", "category": "AI"}
        doc = Document(
            tenant_id="tenant_123",
            title="Test Document",
            metadata=metadata
        )

        assert doc.metadata == metadata

    def test_document_model_dump(self):
        """Test Document can be serialized."""
        doc = Document(
            tenant_id="tenant_123",
            title="Test Document"
        )

        data = doc.model_dump()

        assert data["tenant_id"] == "tenant_123"
        assert data["title"] == "Test Document"
        assert "doc_id" in data

    def test_document_rejects_missing_required_fields(self):
        """Test Document requires tenant_id and title."""
        with pytest.raises(ValidationError):
            Document(title="No Tenant")
        with pytest.raises(ValidationError):
            Document(tenant_id="tenant_123")

    def test_document_rejects_invalid_types(self):
        """Test Document rejects values of the wrong type."""
        with pytest.raises(ValidationError):
            Document(tenant_id=123, title="Test Document")
        with pytest.raises(ValidationError):
            Document(tenant_id="tenant_123", title=None)
        with pytest.raises(ValidationError):
            Document(tenant_id="tenant_123", title="Test", metadata="not a dict")
        with pytest.raises(ValidationError):
            Document(tenant_id="tenant_123", title="Test", created_at="not a date")


class TestSection:
    """Tests for Section model."""

    def test_section_creation(self):
        """Test Section creation with auto-generated section_id."""
        section = Section(
            tenant_id="tenant_123",
            doc_id="doc_456",
            text="This is a test section."
        )

        assert section.tenant_id == "tenant_123"
        assert section.doc_id == "doc_456"
        assert section.text == "This is a test section."
        assert section.section_id is not None
        assert len(section.section_id) > 0
        assert isinstance(section.created_at, datetime)
        assert section.metadata == {}

    def test_section_with_custom_section_id(self):
        """Test Section creation with custom section_id."""
        custom_id = "custom_section_id"
        section = Section(
            tenant_id="tenant_123",
            doc_id="doc_456",
            text="This is a test section.",
            section_id=custom_id
        )

        assert section.section_id == custom_id

    def test_section_with_metadata(self):
        """Test Section creation with metadata."""
        metadata = {"page": 1, "chapter": "Introduction"}
        section = Section(
            tenant_id="tenant_123",
            doc_id="doc_456",
            text="This is a test section.",
            metadata=metadata
        )

        assert section.metadata == metadata

    def test_section_rejects_missing_required_fields(self):
        """Test Section requires tenant_id, doc_id and text."""
        with pytest.raises(ValidationError):
            Section(doc_id="doc_456", text="Text")
        with pytest.raises(ValidationError):
            Section(tenant_id="tenant_123", text="Text")
        with pytest.raises(ValidationError):
            Section(tenant_id="tenant_123", doc_id="doc_456")

    def test_section_rejects_invalid_types(self):
        """Test Section rejects values of the wrong type."""
        with pytest.raises(ValidationError):
            Section(tenant_id="tenant_123", doc_id=456, text="Text")
        with pytest.raises(ValidationError):
            Section(tenant_id="tenant_123", doc_id="doc_456", text=None)
        with pytest.raises(ValidationError):
            Section(tenant_id="tenant_123", doc_id="doc_456", text="Text", metadata=[1, 2])


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            section_id="section_123",
            doc_id="doc_456",
            text="Matched text content.",
            score=0.95,
            metadata={"page": 1}
        )

        assert result.section_id == "section_123"
        assert result.doc_id == "doc_456"
        assert result.text == "Matched text content."
        assert result.score == 0.95
        assert result.metadata == {"page": 1}

    def test_search_result_model_dump(self):
        """Test SearchResult can be serialized."""
        result = SearchResult(
            section_id="section_123",
            doc_id="doc_456",
            text="Matched text content.",
            score=0.95,
            metadata={}
        )

        data = result.model_dump()

        assert data["section_id"] == "section_123"
        assert data["score"] == 0.95

    def test_search_result_rejects_missing_required_fields(self):
        """Test SearchResult requires all of its fields."""
        with pytest.raises(ValidationError):
            SearchResult(
                section_id="section_123",
                doc_id="doc_456",
                text="Matched text content.",
                metadata={}
            )
        with pytest.raises(ValidationError):
            SearchResult(
                section_id="section_123",
                doc_id="doc_456",
                text="Matched text content.",
                score=0.95
            )

    def test_search_result_rejects_invalid_types(self):
        """Test SearchResult rejects values of the wrong type."""
        with pytest.raises(ValidationError):
            SearchResult(
                section_id="section_123",
                doc_id="doc_456",
                text="Matched text content.",
                score="high",
                metadata={}
            )
        with pytest.raises(ValidationError):
            SearchResult(
                section_id="section_123",
                doc_id="doc_456",
                text="Matched text content.",
                score=0.95,
                metadata=None
            )


class TestRaagConfig:
    """Tests for RaagConfig."""

    def test_config_defaults(self):
        """Test RaagConfig has correct default values."""
        config = RaagConfig()

        assert config.embed_batch_size == 100
        assert config.default_k == 10

    def test_config_custom_values(self):
        """Test RaagConfig with custom values."""
        config = RaagConfig(embed_batch_size=50, default_k=5)

        assert config.embed_batch_size == 50
        assert config.default_k == 5
