from pathlib import Path

# BASE PROJECT DIRECTORY

BASE_DIR = Path(__file__).resolve().parent.parent


# DATA DIRECTORIES

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# FILE PATHS

RAW_CORPUS_PATH = RAW_DATA_DIR / "raw_corpus.json"

STRUCTURED_CORPUS_PATH = (
    PROCESSED_DATA_DIR / "structured_corpus.json"
)

CHUNKED_CORPUS_PATH = (
    PROCESSED_DATA_DIR / "chunked_corpus.json"
)

EMBEDDINGS_PATH = (
    PROCESSED_DATA_DIR / "embeddings.npy"
)

FAISS_INDEX_PATH = (
    PROCESSED_DATA_DIR / "faiss_index.bin"
)

# Chunking parameters
CHUNK_SIZE = 120
OVERLAP = 30

# Embedding model name and path
EMBEDDINGS_PATH = (
    PROCESSED_DATA_DIR / "embeddings.npy"
)

# FAISS index path
FAISS_INDEX_PATH = (
    PROCESSED_DATA_DIR / "faiss_index.bin"
)