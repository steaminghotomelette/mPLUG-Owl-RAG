from fastapi import FastAPI
from routes import mplug_chat
from typing import Dict

# --------------------------------------------
# App Initialization
# --------------------------------------------
app = FastAPI(title="LLM Chat Microservice")

# Mount routes
app.include_router(mplug_chat.router)

# Define root endpoint for service health checks
@app.get("/")
async def get_root_status() -> Dict[str, str]:
    return {"status": "LLM chat service is up and running."}