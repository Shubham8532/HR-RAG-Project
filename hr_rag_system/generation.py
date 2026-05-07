from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from hr_rag_system.retrieval import (
    retrieve
)

from hr_rag_system.verification import (
    verify_answer
)


def load_llm():
    """
    Load TinyLlama generation model.
    """

    model_name = (
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return generator, tokenizer


def generate_answer(
    query,
    embedding_model,
    reranker,
    index,
    chunked_corpus,
    generator,
    tokenizer
):
    """
    Generate grounded RAG answer.
    """

    retrieved_docs = retrieve(

        query=query,

        embedding_model=embedding_model,

        reranker=reranker,

        index=index,

        chunked_corpus=chunked_corpus
    )

    context = "\n\n".join([

        doc["text"]

        for doc in retrieved_docs
    ])

    prompt = f"""
You are a medical question answering assistant.

Use ONLY the provided context.

If the answer is not present in the context,
say:

"Not enough information available."

Give a concise answer in 2-3 sentences only.

Context:
{context}

Question:
{query}

Answer:
"""

    response = generator(

        prompt,

        max_new_tokens=50,

        do_sample=False,

        return_full_text=False,

        eos_token_id=tokenizer.eos_token_id
    )

    answer = response[0]["generated_text"]

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

    return {

        "answer": answer,

        "verification_score": round(
            verification_score,
            3
        ),

        "status": status,

        "supporting_context": retrieved_docs[0]["text"]
    }