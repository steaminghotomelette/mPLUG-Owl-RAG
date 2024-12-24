from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from utils.mplug_utils import MplugOwl3ModelManager
from utils.rag_utils import rag_search, create_prompt
from api.routers.rag import rag
from typing import List, Dict
import json
import os
import shutil

# --------------------------------------------
# Initialization
# --------------------------------------------
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

# Model manager
mplug_owl3_manager = MplugOwl3ModelManager("./iic/mPLUG-Owl3-1B-241014")

# --------------------------------------------
# Post Endpoints
# --------------------------------------------
@router.post("/mplug_owl3")
async def chat_with_mplug_owl3(
    files: List[UploadFile] = File(None),
    message_history: str = Form(...),
    prompt: str = Form(...)
) -> Dict:
    """
    Endpoint to interact with the mPLUG-Owl3 model. Handles file uploads and prepares inputs for processing.

    Args:
        files (List[UploadFile]): List of uploaded files (images/videos).
        message_history (str): JSON string representing the message history.

    Returns:
        Dict: Processed response containing the model's output.
    """
    # try:
    # RAG related (limited to text insertion presently)
    context = await rag_search(rag, files, prompt, "CLIP")
    text_data = [document["text"] for document in context["message"]]

    # Reset file pointers after rag_search
    if files:
        for file in files:
            await file.seek(0)

    # Parse the message history and prepare prompt
    parsed_messages = json.loads(message_history)
    rag_prompt = create_prompt(parsed_messages.pop()["content"], text_data)
    parsed_messages.append({"role": "user", "content": rag_prompt})
    parsed_messages.append({"role": "assistant", "content": ""})

    # Ensure temporary directory exists
    temp_dir = "./tmp"
    os.makedirs(temp_dir, exist_ok=True)

    # Prepare lists to store file paths
    image_paths = []
    video_paths = []

    # Process uploaded files
    if files:
        for file in files:
            # Generate a temporary file path
            temp_file_path = os.path.join(temp_dir, file.filename)

            # Save file to temporary directory
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)

            # Categorize file based on MIME type
            if file.content_type.startswith("image/"):
                image_paths.append(temp_file_path)
            elif file.content_type.startswith("video/"):
                video_paths.append(temp_file_path)

    # Generate model response
    response = mplug_owl3_manager.respond(
        messages=parsed_messages, 
        images=image_paths, 
        videos=video_paths
    )

    return {"response": response}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to process files: {str(e)}")
