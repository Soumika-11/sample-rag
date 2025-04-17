from services.rag_service import RAGService

rag_service = RAGService()

# Initialize the RAG service
from utils.startup import initialize_rag_service

initialize_rag_service(rag_service)


def handle_query(question: str):
    return rag_service.query(question)
