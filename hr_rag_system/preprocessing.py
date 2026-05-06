# clean_text()
# structured corpus creation

# chunk_text()
# chunked corpus generation

# =========================================================
# IMPORTS
# =========================================================

import json
import re

from loguru import logger

from hr_rag_system.config import (
    RAW_CORPUS_PATH,
    STRUCTURED_CORPUS_PATH
)


# =========================================================
# CLEAN TEXT FUNCTION
# =========================================================

def clean_text(text):
    """
    Clean noisy dataset text.

    Removes:
    - unknown tokens
    - broken formatting
    - extra whitespace

    Keeps:
    - multilingual text
    - punctuation
    """

    # Remove unknown tokens
    text = text.replace("<unk>", "")

    # Fix broken hyphen formatting
    text = text.replace("@-@", "-")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Fix punctuation spacing
    text = re.sub(r"\s([?.!,])", r"\1", text)

    return text.strip()


# =========================================================
# LOAD RAW CORPUS
# =========================================================

def load_raw_corpus():
    """
    Load raw corpus from JSON.
    """

    logger.info("Loading raw corpus...")

    with open(RAW_CORPUS_PATH, "r", encoding="utf-8") as f:

        raw_corpus = json.load(f)

    logger.success(f"Loaded raw documents: {len(raw_corpus)}")

    return raw_corpus


# =========================================================
# CLEAN + STRUCTURE CORPUS
# =========================================================

def build_structured_corpus(raw_corpus):
    """
    Clean and structure corpus.

    Removes:
    - tiny noisy entries
    - malformed text
    """

    logger.info("Building structured corpus...")

    structured_corpus = []

    for item in raw_corpus:

        text = clean_text(item["text"])

        # Skip tiny/noisy text
        if len(text) > 50:

            structured_corpus.append({

                "id": item["id"],

                "text": text,

                "domain": item["domain"]
            })

    logger.success(
        f"Structured documents: {len(structured_corpus)}"
    )

    return structured_corpus


# =========================================================
# SAVE STRUCTURED CORPUS
# =========================================================

def save_structured_corpus(structured_corpus):
    """
    Save cleaned structured corpus.
    """

    logger.info("Saving structured corpus...")

    with open(
        STRUCTURED_CORPUS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            structured_corpus,
            f,
            ensure_ascii=False,
            indent=2
        )

    logger.success(
        f"Structured corpus saved at: {STRUCTURED_CORPUS_PATH}"
    )


def main():

    raw_corpus = load_raw_corpus()
    structured_corpus = build_structured_corpus(
        raw_corpus
    )
    save_structured_corpus(structured_corpus)



if __name__ == "__main__":

    main()