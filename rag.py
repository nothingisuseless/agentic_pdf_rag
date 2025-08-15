import os
import shutil
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


class RagStore:
    """
    Persistent FAISS-based RAG store.
    - Embeddings: Ollama (nomic-embed-text)
    - Vector store persisted under `index_dir`
    - Single-corpus design: new upload replaces the index (keeps things simple & robust)
    """

    def __init__(self, index_dir: str = "vector_index", embed_model: str = "nomic-embed-text"):
        self.index_dir = index_dir
        self.embed_model = embed_model
        self.embeddings = OllamaEmbeddings(model=self.embed_model)
        self.vs: Optional[FAISS] = None
        self._doc_count: int = 0  # approximate count of chunks

    def try_load(self) -> bool:
        """Load FAISS from disk if it exists."""
        if os.path.isdir(self.index_dir) and os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            try:
                self.vs = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)
                # best-effort: infer count
                self._doc_count = getattr(self.vs, "index", None).ntotal if getattr(self.vs, "index", None) else 0
                return True
            except Exception:
                self.vs = None
        return False

    def is_loaded(self) -> bool:
        return self.vs is not None

    def doc_count(self) -> int:
        return int(self._doc_count or 0)

    def ingest(self, pdf_path: str, replace_index: bool = True):
        """Load a PDF, split, embed, build FAISS, and persist to disk."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        # 1) Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 2) Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # 3) Build FAISS
        vs = FAISS.from_documents(chunks, self.embeddings)

        # 4) Replace existing index dir if requested
        if replace_index and os.path.isdir(self.index_dir):
            try:
                shutil.rmtree(self.index_dir)
            except Exception:
                pass  # ignore if couldn't remove; FAISS might overwrite anyway

        # 5) Persist
        vs.save_local(self.index_dir)

        # 6) Load into memory & update counts
        self.vs = vs
        self._doc_count = getattr(self.vs, "index", None).ntotal if getattr(self.vs, "index", None) else len(chunks)

    def search(self, query: str, k: int = 5) -> List[str]:
        """Return top-k relevant chunks as strings with page citation prefix."""
        if not self.vs:
            raise ValueError("No index loaded. Please upload a PDF first.")

        docs = self.vs.similarity_search(query, k=k)
        results = []
        for d in docs:
            page = d.metadata.get("page", "N/A")
            # Normalize page label
            page_label = f"p. {page}" if isinstance(page, int) else f"{page}"
            results.append(f"[{page_label}] {d.page_content}")
        return results

