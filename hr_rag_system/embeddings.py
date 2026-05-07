# SentenceTransformer loading
# embedding generation
# saving embeddings

import json
import numpy as np
import torch

from loguru import logger

from sentence_transformers import (
    SentenceTransformer
)

from hr_rag_system.config import (
    CHUNKED_CORPUS_PATH,
    EMBEDDINGS_PATH
)


def load_chunked_corpus():
    """
    Load chunked corpus.
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


def prepare_texts(chunked_corpus):
    """
    Extract text from chunks.
    """

    logger.info("Preparing texts...")

    texts = [
        item["text"]
        for item in chunked_corpus
    ]

    logger.success(
        f"Prepared texts: {len(texts)}"
    )

    return texts


def load_embedding_model():
    """
    Load embedding model.
    """

    logger.info("Loading embedding model...")

    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )

    logger.info(
        f"Using device: {device}"
    )
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device
    )

    logger.success("Embedding model loaded.")

    return model


def generate_embeddings(model, texts):
    """
    Generate vector embeddings.
    """

    logger.info("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    logger.success(
        f"Embeddings shape: {embeddings.shape}"
    )

    return embeddings


def save_embeddings(embeddings):
    """
    Save embeddings array.
    """

    logger.info("Saving embeddings...")

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    logger.success(
        f"Embeddings saved at: "
        f"{EMBEDDINGS_PATH}"
    )


def main():

    chunked_corpus = load_chunked_corpus()
    texts = prepare_texts(chunked_corpus)

    model = load_embedding_model()

    embeddings = generate_embeddings(
        model,
        texts
    )

    save_embeddings(embeddings)


if __name__ == "__main__":

    main()