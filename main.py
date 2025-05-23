from fastapi import FastAPI
from routers import query_router

app = FastAPI()

# Include the query router
app.include_router(query_router.router)
