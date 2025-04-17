from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss


class RAGService:
    def __init__(self):
        self.docs_dir = Path("docs")
        self.rag_dir = Path("rag")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.medical_data = []

    def _rag_exists(self):
        return (self.rag_dir / "index.faiss").exists()

    def _create_rag(self):
        print("Creating RAG...")
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in docs directory.")
            return

        texts = []
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file}...")
            reader = PdfReader(pdf_file)
            lines = []
            for page in reader.pages:
                lines.extend(page.extract_text().splitlines())

            for line in lines[2:]:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    code, name = parts
                    self.medical_data.append({"code": code, "name": name.strip()})
                    texts.append(name.strip())

        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        faiss.write_index(self.index, str(self.rag_dir / "index.faiss"))
        print("RAG created and saved.")

    def _load_rag(self):
        print("Loading existing RAG...")
        self.index = faiss.read_index(str(self.rag_dir / "index.faiss"))
        print("RAG loaded.")

    def query(self, question):
        if not self.index:
            return {"error": "RAG not initialized."}

        print("Generating query embedding...")
        query_embedding = self.model.encode([question])

        print("Searching FAISS index...")
        distances, indices = self.index.search(query_embedding, k=5)

        results = []
        for idx in indices[0]:
            if idx < len(self.medical_data):
                results.append(self.medical_data[idx])

        return {"results": results}
