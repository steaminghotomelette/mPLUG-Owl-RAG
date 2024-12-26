import json
import os
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from utils.mplug_manager import MplugOwl3ModelManager
from utils.prompt_utils import filter_stream
from typing import List, Dict, Union

# --------------------------------------------
# Router Initialization
# --------------------------------------------
router = APIRouter(
    prefix="/mplug_owl3",
    tags=["mplug_owl3"]
)

# --------------------------------------------
# Globals and Environment Setup
# --------------------------------------------
load_dotenv()
MPLUG_MODEL_PATH = os.getenv("MPLUG_MODEL_PATH")

# Mplug model manager
mplug_manager = MplugOwl3ModelManager(MPLUG_MODEL_PATH)

# --------------------------------------------
# Post Endpoints
# --------------------------------------------
@router.post("/image_chat", response_model=None) # Response model none is bad practice and should be fixed later
async def image_chat_mplug(
    files: List[UploadFile] = File(...),
    messages: str = Form(...),
    gen_params: str = Form(None),
    query: str = Form(...),
    streaming: bool = Form(False)
) -> Union[Dict[str, str], StreamingResponse]:
    
    # TODO rag search here

    # Parse message history and generation params
    parsed_messages = json.loads(messages)
    parsed_messages.append({"role": "assistant", "content": ""})
    generation_params = json.loads(gen_params)
    sampling = not "num_beams" in gen_params

    # Generate response
    response = mplug_manager.respond(
        parsed_messages,
        images=files,
        gen_params=generation_params,
        sampling=sampling,
        streaming=streaming
    )

    # If streaming is enabled, return a StreamingResponse
    if streaming:
        return StreamingResponse(filter_stream(response), media_type="text/plain")
    else:
        return {"response": response}

@router.post("/video_chat", response_model=None)
async def image_chat_mplug(
    files: List[UploadFile] = File(...),
    messages: str = Form(...),
    gen_params: str = Form(None),
    query: str = Form(...),
    streaming: bool = Form(False)
) -> Union[Dict[str, str], StreamingResponse]:
    
    # TODO rag search here

    # Parse message history and generation params
    parsed_messages = json.loads(messages)
    parsed_messages.append({"role": "assistant", "content": ""})
    generation_params = json.loads(gen_params)
    sampling = not "num_beams" in gen_params

    # Generate response
    response = mplug_manager.respond(
        parsed_messages,
        videos=files,
        gen_params=generation_params,
        sampling=sampling,
        streaming=streaming
    )

    # If streaming is enabled, return a StreamingResponse
    if streaming:
        return StreamingResponse(filter_stream(response), media_type="text/plain")
    else:
        return {"response": response}