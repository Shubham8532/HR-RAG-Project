import json
import faiss
import numpy as np

from loguru import logger

from sentence_transformers import (
    SentenceTransformer
)

from hr_rag_system.config import (
    CHUNKED_CORPUS_PATH,
    FAISS_INDEX_PATH
)


def load_chunked_corpus():
    """
    Load chunked corpus metadata.
    """

    logger.info("Loading chunked corpus...")

    with open(
        CHUNKED_CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        chunked_corpus = json.load(f)

    logger.success(
        f"Loaded chunks: {len(chunked_corpus)}"
    )

    return chunked_corpus


def load_faiss_index():
    """
    Load FAISS vector database.
    """

    logger.info("Loading FAISS index...")

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

    logger.info("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    logger.success(
        "Embedding model loaded."
    )

    return model


def retrieve(
    query,
    model,
    index,
    chunked_corpus,
    top_k=5,
    threshold=1.3
):
    """
    Retrieve relevant documents.
    """

    logger.info(
        f"Searching query: {query}"
    )

    # Query -> embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # FAISS search
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    print("\nDistances:", distances)

    results = []

    for dist, idx in zip(
        distances[0],
        indices[0]
    ):

        if dist < threshold:

            results.append({

                "id":
                chunked_corpus[idx]["id"],

                "source_id":
                chunked_corpus[idx]["source_id"],

                "domain":
                chunked_corpus[idx]["domain"],

                "distance":
                float(dist),

                "text":
                chunked_corpus[idx]["text"]
            })

    # No relevant match
    if len(results) == 0:

        return [{
            "text":
            "Not enough relevant information found.",

            "domain":
            None
        }]

    return results


def main():

    chunked_corpus = load_chunked_corpus()

    index = load_faiss_index()

    model = load_embedding_model()

    query = (
        "blood glucose levels in diabetes"
    )

    results = retrieve(
        query=query,
        model=model,
        index=index,
        chunked_corpus=chunked_corpus
    )

    for r in results:

        print("\n====================")

        print("ID:", r.get("id"))

        print("Source:", r.get("source_id"))

        print("Domain:", r.get("domain"))

        print("Distance:", r.get("distance"))

        print("\nRetrieved Text:\n")

        print(r["text"][:500])


if __name__ == "__main__":

    main()