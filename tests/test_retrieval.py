from src.ask_document.retrieval import reciprocal_rank_fusion


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
