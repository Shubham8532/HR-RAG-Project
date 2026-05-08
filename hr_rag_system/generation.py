import os
from pathlib import Path

from openai import OpenAI

from dotenv import load_dotenv

from hr_rag_system.retrieval import (
    retrieve
)

from hr_rag_system.verification import (
    verify_answer
)

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

client = OpenAI(

    api_key=api_key,

    base_url="https://api.groq.com/openai/v1"
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

    retrieved_docs = retrieve(

        query=query,

        embedding_model=embedding_model,

        reranker=reranker,

        index=index,

        chunked_corpus=chunked_corpus
    )

    # =================================================
    # BUILD CONTEXT
    # =================================================

    context = "\n\n".join([

        doc["text"]

        for doc in retrieved_docs
    ])

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

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0,

            max_tokens=120
        )

        answer = (

            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

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

        return {

            "answer": "No response generated.",

            "verification_score": 0.0,

            "status": "Generation Failed",

            "supporting_context":
            retrieved_docs[0]["text"]
        }

    # =================================================
    # VERIFY ANSWER
    # =================================================

    verification_score = verify_answer(

        answer,

        context,

        embedding_model
    )

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