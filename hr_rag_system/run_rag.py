from hr_rag_system.retrieval import (

    load_chunked_corpus,

    load_faiss_index,

    load_embedding_model,

    load_reranker
)

from hr_rag_system.generation import (
    generate_answer
)

# =====================================================
# MAIN RAG PIPELINE
# =====================================================

def main():
    """
    Run end-to-end RAG pipeline.
    """

    # =================================================
    # LOAD ARTIFACTS + MODELS
    # =================================================

    chunked_corpus = load_chunked_corpus()

    index = load_faiss_index()

    embedding_model = load_embedding_model()

    reranker = load_reranker()

    # =================================================
    # TEST QUERY
    # =================================================

    query = (
        "What are symptoms of diabetes?"
    )

    # =================================================
    # GENERATE ANSWER
    # =================================================

    result = generate_answer(

        query=query,

        embedding_model=embedding_model,

        reranker=reranker,

        index=index,

        chunked_corpus=chunked_corpus
    )

    # =================================================
    # OUTPUT
    # =================================================

    print("\n========== ANSWER ==========\n")

    print(result["answer"])

    print("\n========== VERIFICATION ==========\n")

    print(
        "Grounding Score:",
        result["verification_score"]
    )

    print(
        "Status:",
        result["status"]
    )

    print("\n========== SUPPORTING CONTEXT ==========\n")

    print(
        result["supporting_context"][:1000]
    )


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    main()