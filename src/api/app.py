from fastapi import FastAPI
from api.routers import rag

# Initialize the FastAPI application
app = FastAPI(title="mPLUG-Owl-RAG API")

# Mount the RAG router for handling the RAG-related endpoints
app.include_router(rag.router)

# Define a root endpoint for basic health checks
@app.get("/")
async def read_root():
    """
    Root endpoint to verify the API is running.

    Returns:
        dict: API status message.
    """
    return {"message": "RAG Document Management API is up and running."}
