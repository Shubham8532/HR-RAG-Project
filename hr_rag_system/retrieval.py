import json
import faiss
import numpy as np

from loguru import logger

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from hr_rag_system.config import (
    CHUNKED_CORPUS_PATH,
    FAISS_INDEX_PATH
)


def load_chunked_corpus():
    """
    Load chunked corpus metadata.
    """

    logger.info(
        "Loading chunked corpus..."
    )

    with open(
        CHUNKED_CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        chunked_corpus = json.load(f)

    logger.success(
        f"Loaded chunks: "
        f"{len(chunked_corpus)}"
    )

    return chunked_corpus


def load_faiss_index():
    """
    Load FAISS vector database.
    """

    logger.info(
        "Loading FAISS index..."
    )

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    logger.success(
        f"FAISS vectors loaded: "
        f"{index.ntotal}"
    )

    return index


def load_embedding_model():
    """
    Load embedding model.
    """

    logger.info(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    logger.success(
        "Embedding model loaded."
    )

    return model


def load_reranker():
    """
    Load CrossEncoder reranker.
    """

    logger.info(
        "Loading reranker..."
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    logger.success(
        "Reranker loaded."
    )

    return reranker


def retrieve(
    query,
    embedding_model,
    reranker,
    index,
    chunked_corpus,
    top_k=3
):
    """
    Retrieve and rerank documents.
    """

    logger.info(
        f"Searching query: {query}"
    )

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        20
    )

    logger.info(
        f"FAISS distances: {distances}"
    )

    candidates = []

    for idx in indices[0]:

        candidates.append(
            chunked_corpus[idx]
        )

    pairs = []

    for candidate in candidates:

        pairs.append([
            query,
            candidate["text"]
        ])

    scores = reranker.predict(pairs)

    reranked_results = []

    for score, candidate in zip(
        scores,
        candidates
    ):

        reranked_results.append({

            "score": float(score),

            "text": candidate["text"]
        })

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["score"],
        reverse=True
    )

    return reranked_results[:top_k]