import pytest
from pydantic import ValidationError
from rag_app.models import DocumentMetadata, Document


def test_document_metadata_valid_date():
    valid_metadata = {
        "date": "2024-10",
        "url": "https://example.com",
        "title": "Test Title",
    }

    try:
        DocumentMetadata(**valid_metadata)
    except ValidationError as e:
        pytest.fail(f"Unexpected ValidationError: {e}")


def test_document_metadata_invalid_date():
    invalid_metadata = {
        "date": "2024-13",  # Invalid month
        "url": "https://example.com",
        "title": "Test Title",
    }
    with pytest.raises(ValueError) as excinfo:
        DocumentMetadata(**invalid_metadata)
    assert "Date format must be YYYY-MM" in str(excinfo.value)


def test_document_creation():
    document_data = {
        "id": "doc123",
        "content": "This is a test document.",
        "filename": "test_document.txt",
        "metadata": {
            "date": "2024-10",
            "url": "https://example.com",
            "title": "Test Title",
        },
    }
    try:
        Document(**document_data)
    except ValidationError as e:
        pytest.fail(f"Unexpected ValidationError: {e}")


def test_document_creation_with_invalid_metadata():
    document_data = {
        "id": "doc123",
        "content": "This is a test document.",
        "filename": "test_document.txt",
        "metadata": {
            "date": "2024-13",  # Invalid month
            "url": "https://example.com",
            "title": "Test Title",
        },
    }
    with pytest.raises(ValueError) as excinfo:
        Document(**document_data)
    assert "Date format must be YYYY-MM" in str(excinfo.value)
