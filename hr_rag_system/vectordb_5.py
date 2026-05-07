import faiss
import numpy as np

from loguru import logger

from hr_rag_system.config import (
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH
)


def load_embeddings():
    """
    Load embeddings array.
    """

    logger.info("Loading embeddings...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    logger.success(
        f"Embeddings loaded: "
        f"{embeddings.shape}"
    )

    return embeddings


def build_faiss_index(embeddings):
    """
    Build FAISS vector index.
    """

    logger.info("Building FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(embeddings)
    )

    logger.success(
        f"FAISS index size: "
        f"{index.ntotal}"
    )

    return index


def save_faiss_index(index):
    """
    Save FAISS index.
    """

    logger.info("Saving FAISS index...")

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

    logger.success(
        f"FAISS index saved at: "
        f"{FAISS_INDEX_PATH}"
    )


def main():

    embeddings = load_embeddings()

    index = build_faiss_index(
        embeddings
    )

    save_faiss_index(index)


if __name__ == "__main__":

    main()