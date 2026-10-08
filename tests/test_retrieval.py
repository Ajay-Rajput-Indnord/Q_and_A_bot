from src.ask_document.retrieval import reciprocal_rank_fusion
from src.ask_document.bm25 import content_tokens, tokenize


def result(chunk_id: str, retriever: str) -> dict:
    return {"chunk_id": chunk_id, "text": chunk_id, "metadata": {}, "score": 1.0, "retriever": retriever}


def test_rrf_prioritizes_chunks_present_in_both_rankings():
    dense = [result("shared", "dense"), result("dense-only", "dense")]
    bm25 = [result("shared", "bm25"), result("bm25-only", "bm25")]

    fused = reciprocal_rank_fusion([dense, bm25], rrf_k=60, top_k=3)

    assert fused[0]["chunk_id"] == "shared"
    assert fused[0]["retriever"] == "hybrid_rrf"


def test_rrf_returns_requested_number_of_results():
    results = [[result(f"chunk-{i}", "dense") for i in range(5)]]

    fused = reciprocal_rank_fusion(results, top_k=2)

    assert len(fused) == 2


def test_tokenize_preserves_technical_compounds():
    tokens = tokenize("text-embedding-3-small and mo.ui.text v1.2.3")

    assert "text-embedding-3-small" in tokens
    assert "mo.ui.text" in tokens
    assert "v1.2.3" in tokens


def test_content_tokens_removes_question_filler_but_keeps_subject():
    tokens = content_tokens("What are the parameters for mo.ui.text?")

    assert "what" not in tokens
    assert "parameters" in tokens
    assert "mo.ui.text" in tokens
