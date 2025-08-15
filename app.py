import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import requests

from rag import RagStore

# ---------------- App Setup ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "uploads"
INDEX_DIR = "vector_index"   # persistent FAISS index directory
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

OLLAMA_BASE = "http://127.0.0.1:11434/api"

# Initialize RAG store (loads existing FAISS if present)
rag = RagStore(index_dir=INDEX_DIR)

def allowed_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ollama": True, 
        "index_loaded": rag.is_loaded(),
        "docs_count": rag.doc_count()
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """
    Return available chat-capable models (filters out embedding models like 'nomic-embed-text')
    as [{ "name": "llama3:latest" }, ...]
    """
    try:
        r = requests.get(f"{OLLAMA_BASE}/tags", timeout=8)
        r.raise_for_status()
        data = r.json()
        out = []
        for m in data.get("models", []):
            name = m.get("name", "")
            # filter out embedding models by name
            if "embed" in name.lower():
                continue
            # optionally filter families that are not text LLMs (keep simple & permissive)
            out.append({"name": name})
        return jsonify(out)
    except Exception:
        # Return an array (frontend expects it). Show empty if Ollama isn't reachable.
        return jsonify([]), 200


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """
    Upload and ingest a PDF into persistent FAISS.
    If an index already exists, we replace it (single-corpus design).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        f.save(path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {e}"}), 500

    try:
        # Ingest and persist (replace existing index)
        rag.ingest(pdf_path=path, replace_index=True)
        return jsonify({"message": "PDF ingested successfully and index saved!"})
    except Exception as e:
        return jsonify({"error": f"Ingestion failed: {e}"}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    model = (data.get("model") or "").strip()
    try:
        temperature = float(data.get("temperature", 0.2))
    except Exception:
        temperature = 0.2
    temperature = max(0.0, min(1.0, temperature))

    if not question:
        return jsonify({"error": "Question is required"}), 400
    if not model:
        return jsonify({"error": "Model is required"}), 400
    if not rag.is_loaded():
        return jsonify({"error": "No document index loaded. Please upload a PDF first."}), 400

    # Retrieve top chunks (citations included)
    try:
        top_snippets = rag.search(question, k=5)  # list[str] with [page X] prefix
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {e}"}), 500

    context_block = "\n\n".join(top_snippets)

    # Agentic-style instruction to reason stepwise and cite pages
    system_prompt = (
        "You are an expert assistant. Use ONLY the provided document context to answer the user’s question.\n"
        "Think step by step, extract the relevant facts, and provide a concise final answer.\n"
        "CITE page numbers in square brackets where evidence appears (e.g., [p. 12]).\n"
        "If the answer is not in the context, say so explicitly.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/generate",
            json={
                "model": model,
                "prompt": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                    # IMPORTANT: do NOT include generation-only options when calling embeddings.
                }
            },
            timeout=120
        )
        if resp.status_code != 200:
            return jsonify({"error": f"Ollama error: {resp.text}"}), 502

        payload = resp.json()
        answer = payload.get("response") or payload.get("output") or ""
        return jsonify({"answer": answer.strip() or "(empty response)"}), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to contact Ollama: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500


# --------------- Main ----------------
if __name__ == "__main__":
    # On startup, try to load existing FAISS index (if any)
    rag.try_load()
    app.run(host="0.0.0.0", port=5000, debug=True)

