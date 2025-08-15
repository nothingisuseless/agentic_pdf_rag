# 📄 Ollama PDF-based RAG with Agentic AI

An end-to-end **Retrieval-Augmented Generation (RAG)** application built with **Python (Flask)**, **Ollama**, and **LangChain**, enabling you to upload PDFs, ingest their content, and ask contextual questions with **Agentic AI capabilities**.

## ✨ Features
- **PDF Upload & Ingestion** – Extracts text from uploaded PDFs and stores embeddings.
- **Ollama LLM Integration** – Uses local models like `llama3` for answering questions.
- **Embeddings with nomic-embed-text** – For efficient vector search.
- **Agentic AI** – The LLM can decide when and how to use tools to fetch answers.
- **Frontend UI** – Simple HTML + JavaScript with model selection, temperature control, and PDF upload.
- **Lightweight** – No external DB required (uses FAISS locally).

---

## 🚀 Getting Started

### Prerequisites
- Python **3.10+**
- [Ollama](https://ollama.ai) installed and running locally.
- (Optional) GPU support for faster performance.

---

### Install Dependencies
```bash
git clone https://github.com/nothingisuseless/agentic_pdf_rag.git
cd agentic_pdf_rag

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```
---

### **Pull Required Ollama Models**
```bash
ollama pull llama3
ollama pull nomic-embed-tex
```
---

### Start Ollama in a Separate Terminal
```
ollama serve
```
---

### Run the Application
```
python app.py

```
The app will run at:

```
http://127.0.0.1:5000
```
---
### Project Structure

```
📁 agentic_pdf_rag/
 ├── app.py              # Flask backend API
 ├── rag.py              # RAG pipeline logic
 ├── templates/
 │    └── index.html     # Frontend HTML
 ├── static/
 │    ├── app.js         # Frontend JavaScript
 │    └── style.css      # Custom CSS (Bluish Light Mode)
 ├── requirements.txt    # Python dependencies
 └── README.md           # Documentation
```

---

### 🛠 How to Use

1. Open the web app in your browser.
2. Select an Ollama model (e.g., llama3).
3. Upload a PDF file for ingestion.
4. Ask any question related to the document.
5. The AI will provide contextual answers using retrieved data.

---

### ⚡ Agentic AI Benefits

1. Decides when retrieval is needed.
2. Combines retrieved data with prior knowledge.
3. Can be extended to call APIs, search web data, or use tools.

---

### 🐞 Troubleshooting

| Issue                   | Solution                                                                   |
| ----------------------- | -------------------------------------------------------------------------- |
| **Model not found**     | Run `ollama pull <model-name>`                                             |
| **Port already in use** | Stop Ollama and restart: `pkill ollama && ollama serve`                    |
| **Timeout errors**      | Increase timeout in `app.py` or warm up model: `ollama run llama3 "Hello"` |

---

