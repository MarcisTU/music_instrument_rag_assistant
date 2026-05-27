from typing import List

from src.models.schemas import RAGDocument


def reciprocal_rank_fusion(
        bm25_results: list[RAGDocument],
        vector_results: list[RAGDocument],
        k: int = 60
) -> List[RAGDocument]:
    """
    Combines BM25 and Vector search results of RAGDocument objects using RRF.

    Args:
        bm25_results: List of RAGDocument objects from keyword search (ordered best to worst)
        vector_results: List of RAGDocument objects from vector search (ordered best to worst)
        k: Smoothing factor

    Returns:
        A list of RAGDocument sorted by highest score to lowest.
    """

    rrf_registry = {}

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.product_id
        if doc_id not in rrf_registry:
            rrf_registry[doc_id] = {"score": 0.0, "model": doc}

        rrf_registry[doc_id]["score"] += 1.0 / (k + rank)

    # Score Vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.product_id
        if doc_id not in rrf_registry:
            rrf_registry[doc_id] = {"score": 0.0, "model": doc}

        rrf_registry[doc_id]["score"] += 1.0 / (k + rank)

    # Sort dictionary items based on the nested "score" key in descending order
    sorted_items = sorted(
        rrf_registry.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    return [item["model"] for item in sorted_items]
