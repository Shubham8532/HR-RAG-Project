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
from hr_rag_system.config import (MEDICAL_TOPICS_PATH)

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
# LOAD LAVITA MEDICAL QA DATASET
# =========================================================

def load_lavita_medical(limit=3000):
    """
    Load instruction-style medical QA corpus.

    Dataset:
    - lavita/medical-qa-datasets

    Returns:
        List[dict]
    """

    logger.info(
        "Loading Lavita medical QA dataset..."
    )

    dataset = load_dataset(
        "lavita/medical-qa-datasets",
        "all-processed",
        split=f"train[:{limit}]"
    )

    data = []

    for i, item in enumerate(dataset):

        text = (
            item["instruction"] + " " +
            item["input"] + " " +
            item["output"]
        )

        data.append({

            "id": f"lavita_{i}",

            "text": text,

            "domain": "medical_instruction_qa"
        })

    logger.success(
        f"Lavita dataset loaded: {len(data)} documents"
    )

    return data

# =========================================================
# LOAD MEDICAL QA DATASET
# =========================================================

def load_medical_qa(limit=3000):
    """
    Load educational medical QA corpus.

    Dataset:
    - Starlord1010/Medical-QA-dataset

    Returns:
        List[dict]
    """

    logger.info(
        "Loading medical QA dataset..."
    )

    dataset = load_dataset(
        "Starlord1010/Medical-QA-dataset",
        split=f"train[:{limit}]"
    )

    data = []

    for i, item in enumerate(dataset):

        text = (
            item["question"] + " " +
            item["response"]
        )

        data.append({

            "id": f"medqa_{i}",

            "text": text,

            "domain": "medical_qa"
        })

    logger.success(
        f"Medical QA dataset loaded: {len(data)} documents"
    )

    return data

# =========================================================
# LOAD MEDICAL WIKIPEDIA ARTICLES
# =========================================================

def load_medical_wikipedia():
    """
    Load medical Wikipedia articles
    using topics from text file.

    Returns:
        List[dict]
    """

    logger.info(
        "Loading medical Wikipedia articles..."
    )

    with open(
        MEDICAL_TOPICS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        medical_topics = [

            line.strip()

            for line in f

            if line.strip()
        ]

    data = []

    for i, topic in enumerate(medical_topics):

        try:

            logger.info(
                f"Loading topic: {topic}"
            )

            page = wikipedia.page(
                topic,
                auto_suggest=False
            )

            data.append({

                "id": f"medwiki_{i}",

                "text": page.content[:5000],

                "domain": "medical_wikipedia"
            })

            time.sleep(
                random.uniform(3, 4)
            )

        except Exception as e:

            logger.warning(
                f"Skipped topic: {topic}"
            )

            logger.warning(str(e))

            continue

    logger.success(
        f"Medical Wikipedia loaded: {len(data)} documents"
    )

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
    corpus.extend(load_medical_qa())
    corpus.extend(load_lavita_medical())
    corpus.extend(load_medical_wikipedia())
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