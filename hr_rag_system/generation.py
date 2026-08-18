import os
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from hr_rag_system.retrieval import retrieve
from hr_rag_system.verification import verify_answer


# =====================================================
# LOAD ENV VARIABLES
# =====================================================

_env_path = (
    Path(__file__).resolve().parent.parent / ".env"
)

load_dotenv(dotenv_path=_env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise EnvironmentError(
        f"GROQ_API_KEY not found. "
        f"Looked for .env at: {_env_path}"
    )


# =====================================================
# GROQ CLIENT
# =====================================================

client = Groq(
    api_key=api_key
)


# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(
    query,
    embedding_model,
    reranker,
    index,
    chunked_corpus
):
    """
    Generate grounded RAG answer
    using Groq API.
    """

    # =================================================
    # RETRIEVE DOCUMENTS
    # =================================================

    print("\n[1/5] Retrieving documents...")

    start_time = time.time()

    retrieved_docs = retrieve(
        query=query,
        embedding_model=embedding_model,
        reranker=reranker,
        index=index,
        chunked_corpus=chunked_corpus
    )

    print(
        f"[1/5] Retrieval completed in "
        f"{time.time() - start_time:.2f}s"
    )

    # =================================================
    # SAFETY CHECK
    # =================================================

    if not retrieved_docs:

        return {
            "answer": "Not enough information available.",
            "verification_score": 0.0,
            "status": "No relevant context",
            "supporting_context": "No supporting context found."
        }

    # =================================================
    # BUILD CONTEXT
    # =================================================

    print("[2/5] Building context...")

    context = "\n\n".join(
        doc["text"]
        for doc in retrieved_docs
    )

    print(
        f"[2/5] Context length: "
        f"{len(context)} characters"
    )

    # =================================================
    # BUILD PROMPT
    # =================================================

    prompt = f"""
You are a medical question answering assistant.

Use ONLY the provided context.

If the answer is not present in the context,
say:

"Not enough information available."

Give a concise answer in 2-3 complete sentences only.

Do not leave incomplete sentences.

Context:
{context}

Question:
{query}

Answer:
"""

    # =================================================
    # GROQ GENERATION
    # =================================================

    try:

        print("[3/5] Calling Groq LLM...")

        start_time = time.time()

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_completion_tokens=300
        )

        print(
            f"[3/5] LLM completed in "
            f"{time.time() - start_time:.2f}s"
        )

        # =================================================
        # EXTRACT ANSWER
        # =================================================

        message = response.choices[0].message

        answer = (message.content or "").strip()

        print(
            f"[3/5] Answer length: "
            f"{len(answer)} characters"
        )

    except Exception as e:

        print(
            f"[3/5] LLM ERROR: {str(e)}"
        )

        return {

            "answer":
            f"LLM generation failed:\n\n{str(e)}",

            "verification_score": 0.0,

            "status": "Generation Failed",

            "supporting_context":
            retrieved_docs[0]["text"]
        }

    # =================================================
    # EMPTY ANSWER SAFETY
    # =================================================

    if not answer:

        print("[3/5] LLM returned an empty answer.")

        return {

            "answer": "No response generated.",

            "verification_score": 0.0,

            "status": "Generation Failed",

            "supporting_context":
            retrieved_docs[0]["text"]
        }

    print("[4/5] Answer generated successfully.")

    # =================================================
    # VERIFY ANSWER
    # =================================================

    try:

        print("[5/5] Verifying answer...")

        start_time = time.time()

        verification_score = verify_answer(
            answer,
            context,
            embedding_model
        )

        print(
            f"[5/5] Verification completed in "
            f"{time.time() - start_time:.2f}s"
        )

    except Exception as e:

        print(
            f"[5/5] Verification ERROR: {str(e)}"
        )

        return {

            "answer": answer,

            "verification_score": 0.0,

            "status": "Verification Failed",

            "supporting_context":
            retrieved_docs[0]["text"]
        }

    # =================================================
    # DETERMINE STATUS
    # =================================================

    if verification_score >= 0.80:

        status = "Strongly grounded"

    elif verification_score >= 0.65:

        status = "Partially grounded"

    else:

        status = "Possible hallucination"

    # =================================================
    # RETURN RESULT
    # =================================================

    return {

        "answer": answer,

        "verification_score": round(
            verification_score,
            3
        ),

        "status": status,

        "supporting_context":
        retrieved_docs[0]["text"]
    }