from fastapi import FastAPI
from api.routers import rag, chat

# Initialize the FastAPI application
app = FastAPI(title="mPLUG-Owl-RAG API")

# Mount routers
app.include_router(rag.router)
app.include_router(chat.router)

# Define a root endpoint for basic health checks
@app.get("/")
async def read_root():
    """
    Root endpoint to verify that the API is running.

    Returns:
        dict: API status message.
    """
    return {"message": "RAG VQA API is up and running."}
