from fastapi import APIRouter, Depends
from controllers.query_controller import handle_query
from middlewares.auth_middleware import verify_api_key

router = APIRouter()


@router.get("/query")
def query_rag(question: str, api_key: str = Depends(verify_api_key)):
    return handle_query(question)
