# dataset loading
# raw corpus generation
# saving raw corpus

# =========================================================
# IMPORTS
# =========================================================

from datasets import load_dataset
import json
from loguru import logger

from hr_rag_system.config import RAW_CORPUS_PATH

# =========================================================
# LOAD WIKITEXT DATASET
# =========================================================

def load_wikitext(limit=5000):
    """
    Load history/general knowledge corpus.
    Dataset:
    - Wikitext
    Returns:
        List[dict]
    """

    logger.info("Loading Wikitext dataset...")

    dataset = load_dataset(
        "wikitext",
        "wikitext-103-v1",
        split=f"train[:{limit}]"
    )

    data = []

    for i, item in enumerate(dataset):

        text = item["text"]

        data.append({
            "id": f"wiki_{i}",
            "text": text,
            "domain": "history"
        })

    logger.success(f"Wikitext loaded: {len(data)} documents")

    return data


# =========================================================
# LOAD SCIENCE DATASET
# =========================================================

def load_science(limit=3000):
    """
    Load scientific/news dataset.
    Dataset:
    - AG News
    Returns:
        List[dict]
    """

    logger.info("Loading scientific dataset...")

    dataset = load_dataset(
        "ag_news",
        split=f"train[:{limit}]"
    )

    data = []

    for i, item in enumerate(dataset):

        text = item["text"]

        data.append({
            "id": f"sci_{i}",
            "text": text,
            "domain": "scientific"
        })

    logger.success(f"Scientific dataset loaded: {len(data)} documents")

    return data


# =========================================================
# LOAD MEDICAL DATASET
# =========================================================

def load_medical(limit=3000):
    """
    Load biomedical research dataset.
    Dataset:
    - PubMedQA
    Returns:
        List[dict]
    """

    logger.info("Loading medical dataset...")

    dataset = load_dataset(
        "pubmed_qa",
        "pqa_labeled",
        split="train"
    )

    data = []

    for i, item in enumerate(dataset):

        # Combine question + biomedical context
        text = (
            item["question"] + " " +
            " ".join(item["context"]["contexts"])
        )

        data.append({
            "id": f"med_{i}",
            "text": text,
            "domain": "medical"
        })

        if i >= limit:
            break

    logger.success(f"Medical dataset loaded: {len(data)} documents")

    return data

# =========================================================
# BUILD RAW CORPUS
# =========================================================

def build_raw_corpus():
    """
    Combine all datasets into one raw corpus.

    Returns:
        List[dict]
    """

    logger.info("Building raw corpus...")

    corpus = []
    corpus.extend(load_wikitext())
    corpus.extend(load_science())
    corpus.extend(load_medical())
    logger.success(f"Total raw documents: {len(corpus)}")

    return corpus


# =========================================================
# SAVE RAW CORPUS
# =========================================================

def save_raw_corpus(corpus):
    """
    Save raw corpus to JSON file.
    """

    logger.info("Saving raw corpus...")

    with open(RAW_CORPUS_PATH, "w", encoding="utf-8") as f:

        json.dump(
            corpus,
            f,
            ensure_ascii=False,
            indent=2
        )

    logger.success(f"Raw corpus saved at: {RAW_CORPUS_PATH}")


def main():
    corpus = build_raw_corpus()
    save_raw_corpus(corpus)


if __name__ == "__main__":

    main()