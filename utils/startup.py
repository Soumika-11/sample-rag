from pathlib import Path
from services.rag_service import RAGService


def initialize_rag_service(rag_service: RAGService):
    docs_dir = Path("docs")
    rag_dir = Path("rag")
    docs_dir.mkdir(exist_ok=True)
    rag_dir.mkdir(exist_ok=True)
    if not rag_service._rag_exists():
        rag_service._create_rag()
    else:
        rag_service._load_rag()
