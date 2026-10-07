import pandas as pd
import pytest

from src.ask_document.ingestion import create_chunks, load_documents


def test_load_documents_validates_required_columns(tmp_path):
    path = tmp_path / "documents.csv"
    pd.DataFrame({"index": [0], "source_url": ["https://example.com"], "text": ["A document."]}).to_csv(path, index=False)

    frame = load_documents(path)

    assert list(frame.columns) == ["index", "source_url", "text"]
    assert frame.loc[0, "index"] == 0


def test_load_documents_rejects_duplicate_indices(tmp_path):
    path = tmp_path / "documents.csv"
    pd.DataFrame({"index": [0, 0], "source_url": ["a", "b"], "text": ["one", "two"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_documents(path)


def test_create_chunks_adds_stable_metadata():
    frame = pd.DataFrame({"index": [3], "source_url": ["https://example.com"], "text": ["A document with useful information."]})

    chunks = create_chunks(frame)

    assert chunks[0]["document_index"] == 3
    assert chunks[0]["chunk_id"] == "doc-03-chunk-0000"
    assert chunks[0]["source_url"] == "https://example.com"
