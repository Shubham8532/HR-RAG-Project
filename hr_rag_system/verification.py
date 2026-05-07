from sklearn.metrics.pairwise import (
    cosine_similarity
)


def verify_answer(
    answer,
    context,
    embedding_model
):
    """
    Verify grounding score between
    generated answer and context.
    """

    answer_embedding = embedding_model.encode(
        [answer],
        convert_to_numpy=True
    )

    context_embedding = embedding_model.encode(
        [context],
        convert_to_numpy=True
    )

    score = cosine_similarity(
        answer_embedding,
        context_embedding
    )[0][0]

    return float(score)