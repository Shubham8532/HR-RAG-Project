from flask import Flask
from flask import render_template
from flask import request

from hr_rag_system.retrieval import (
    load_chunked_corpus,
    load_faiss_index,
    load_embedding_model,
    retrieve
)


# =========================================================
# LOAD RAG COMPONENTS
# =========================================================

print("Loading RAG system...")

chunked_corpus = load_chunked_corpus()

index = load_faiss_index()

model = load_embedding_model()

print("RAG system loaded.")


# =========================================================
# CREATE FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = []

    query = ""

    if request.method == "POST":

        query = request.form["query"]

        results = retrieve(
            query=query,
            model=model,
            index=index,
            chunked_corpus=chunked_corpus,
            top_k=5,
            threshold=1.3
        )

    return render_template(
        "index.html",
        query=query,
        results=results
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)