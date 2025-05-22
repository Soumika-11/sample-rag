from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import json


class RAGService:
    def __init__(self):
        self.docs_dir = Path("docs")
        self.rag_dir = Path("rag")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.medical_data = []

    def _rag_exists(self):
        index_exists = (self.rag_dir / "index.faiss").exists()
        data_exists = (self.rag_dir / "medical_data.json").exists()
        return index_exists and data_exists

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
                # Expecting line format: ICD10_CODE HCC_CODE Description
                parts = line.split(" ", 2)
                if len(parts) == 3:
                    icd10_code, hcc_code, name = parts
                    self.medical_data.append({
                        "icd10_code": icd10_code,
                        "hcc_code": hcc_code,
                        "name": name.strip()
                    })
                    texts.append(name.strip())
                elif len(parts) == 2:
                    code, name = parts
                    self.medical_data.append({
                        "icd10_code": code,
                        "hcc_code": None,
                        "name": name.strip()
                    })
                    texts.append(name.strip())

        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        # Save medical data to JSON
        with open(self.rag_dir / "medical_data.json", "w") as f:
            json.dump(self.medical_data, f)

        faiss.write_index(self.index, str(self.rag_dir / "index.faiss"))
        print("RAG created and saved.")

    def _load_rag(self):
        print("Loading existing RAG...")
        self.index = faiss.read_index(str(self.rag_dir / "index.faiss"))

        # Load medical data from JSON
        with open(self.rag_dir / "medical_data.json", "r") as f:
            self.medical_data = json.load(f)

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
                entry = self.medical_data[idx]
                results.append({
                    "icd10_code": entry.get("icd10_code"),
                    "hcc_code": entry.get("hcc_code"),
                    "name": entry.get("name")
                })

        return {"results": results}
