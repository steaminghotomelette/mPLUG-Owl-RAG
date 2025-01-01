from fastapi import FastAPI
from routes import rag
from typing import Dict

# --------------------------------------------
# App Initialization
# --------------------------------------------
app = FastAPI(title="RAG Microservice")

# Mount routes
app.include_router(rag.router)

# Define root endpoint for service health checks
@app.get("/")
async def get_root_status() -> Dict[str, str]:
    return {"status": "RAG service is up and running."}