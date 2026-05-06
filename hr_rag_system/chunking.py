import json

from loguru import logger

from hr_rag_system.config import (
    STRUCTURED_CORPUS_PATH,
    CHUNKED_CORPUS_PATH,
    CHUNK_SIZE,
    OVERLAP
)


def load_structured_corpus():
    """
    Load cleaned structured corpus.
    """

    logger.info("Loading structured corpus...")

    with open(
        STRUCTURED_CORPUS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        structured_corpus = json.load(f)

    logger.success(
        f"Loaded documents: {len(structured_corpus)}"
    )

    return structured_corpus


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP
):
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        chunks.append(
            " ".join(chunk)
        )

        start += chunk_size - overlap

    return chunks


def build_chunked_corpus(structured_corpus):
    """
    Create chunked corpus.
    """

    logger.info("Creating chunks...")

    chunked_corpus = []

    chunk_id = 0

    for item in structured_corpus:

        chunks = chunk_text(item["text"])

        for chunk in chunks:

            if len(chunk) > 50:

                chunked_corpus.append({

                    "id": f"chunk_{chunk_id}",

                    "source_id": item["id"],

                    "domain": item["domain"],

                    "text": chunk
                })

                chunk_id += 1

    logger.success(
        f"Total chunks: {len(chunked_corpus)}"
    )

    return chunked_corpus


def save_chunked_corpus(chunked_corpus):
    """
    Save chunked corpus.
    """

    logger.info("Saving chunked corpus...")

    with open(
        CHUNKED_CORPUS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunked_corpus,
            f,
            ensure_ascii=False,
            indent=2
        )

    logger.success(
        f"Chunked corpus saved at: "
        f"{CHUNKED_CORPUS_PATH}"
    )


def main():

    structured_corpus = load_structured_corpus()

    chunked_corpus = build_chunked_corpus(
        structured_corpus
    )

    save_chunked_corpus(chunked_corpus)


if __name__ == "__main__":

    main()