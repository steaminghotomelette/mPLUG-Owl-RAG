from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from utils.mplug_utils import MplugOwl3ModelManager
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
    message_history: str = Form(...)
) -> Dict:
    """
    Endpoint to interact with the mPLUG-Owl3 model. Handles file uploads and prepares inputs for processing.

    Args:
        files (List[UploadFile]): List of uploaded files (images/videos).
        message_history (str): JSON string representing the message history.

    Returns:
        Dict: Processed response containing the model's output.
    """
    try:
        # Parse the message history
        parsed_messages = json.loads(message_history)
        parsed_messages.append({"role": "assistant", "content": ""})  # Placeholder for model response

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process files: {str(e)}")
