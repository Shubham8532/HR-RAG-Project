from flask import (
    Flask,
    render_template,
    request
)

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
# LOAD MODELS + ARTIFACTS ONCE AT STARTUP
# =====================================================

print("Loading RAG system...")

chunked_corpus    = load_chunked_corpus()
index             = load_faiss_index()
embedding_model   = load_embedding_model()
reranker          = load_reranker()

print("RAG system ready.")

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)

# =====================================================
# HOME PAGE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error  = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()

        if query:
            try:
                result = generate_answer(
                    query           = query,
                    embedding_model = embedding_model,
                    reranker        = reranker,
                    index           = index,
                    chunked_corpus  = chunked_corpus
                )
            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        result = result,
        error  = error
    )

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    app.run(
        host  = "0.0.0.0",
        port  = 5000,
        debug = False
    )