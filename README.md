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

### 1️⃣ Prerequisites
- Python **3.10+**
- [Ollama](https://ollama.ai) installed and running locally.
- (Optional) GPU support for faster performance.

---

### Install Dependencies
```bash
git clone https://github.com/nothingisuseless/agentic_pdf_rag.git
cd YOUR_REPO_NAME

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt


### **Install Dependencies**
```bash
