import json
import os
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from utils.mplug_manager import MplugOwl3ModelManager
from utils.prompt_utils import filter_stream, create_system_prompt, create_user_prompt
from requests import post
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
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION")
RAG_API_BASE_URL = "http://127.0.0.1:8080/rag_documents"

# Mplug model manager
mplug_manager = MplugOwl3ModelManager(MPLUG_MODEL_PATH, ATTN_IMPLEMENTATION)

# --------------------------------------------
# Request Functions
# --------------------------------------------
def search_rag_image(files_upload: List, query: str, embed_model: str, domain: str) -> dict:
    try:
        url = f"{RAG_API_BASE_URL}/search"
        files = []
        for file in files_upload:
            files.append(("files", (file.name, file.read(), file.type)))

        data = {"query": query, "embedding_model": embed_model, "domain": domain}
        response = post(url, files=files, data=data)
        return response.json()
    except Exception as e:
        raise Exception(f"Search RAG failed: {e}")

def search_rag_video(files_upload: List, query: str, embed_model: str, domain: str) -> dict:
    try:
        url = f"{RAG_API_BASE_URL}/search_video"
        files = []
        for file in files_upload:
            files.append(("files", (file.name, file.read(), file.type)))

        data = {"query": query, "embedding_model": embed_model, "domain": domain}
        response = post(url, files=files, data=data)
        return response.json()
    except Exception as e:
        raise Exception(f"Search RAG failed: {e}")

# --------------------------------------------
# Post Endpoints
# --------------------------------------------
@router.post("/image_chat", response_model=None) # Response model none is bad practice and should be fixed later
async def image_chat_mplug(
    files: List[UploadFile] = File(...),
    messages: str = Form(...),
    gen_params: str = Form(None),
    domain: str = Form(...),
    embedding_model: str = Form(...),
    query: str = Form(...),
    streaming: bool = Form(False)
) -> Union[Dict[str, str], StreamingResponse]:
    
    context = await search_rag_image(files, query, embedding_model, domain)
    text_data = [document["text"] for document in context["message"]]

    # Parse message history and generation params
    parsed_messages = json.loads(messages)
    generation_params = json.loads(gen_params)
    sampling = not "num_beams" in gen_params
    
    # Create system prompt only for first interaction
    if len(parsed_messages) == 0 or parsed_messages[0]["role"] != "system":
        system_prompt = create_system_prompt(domain)
        parsed_messages.insert(0, {"role": "system", "content": system_prompt})

    # Create user query prompt for the current interaction
    user_query = parsed_messages.pop()["content"]
    rag_prompt = create_user_prompt(user_query, text_data)

    parsed_messages.append({"role": "user", "content": rag_prompt})
    parsed_messages.append({"role": "assistant", "content": ""})

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
    domain: str = Form(...),
    embedding_model: str = Form(...),
    query: str = Form(...),
    streaming: bool = Form(False)
) -> Union[Dict[str, str], StreamingResponse]:
    
    context = await search_rag_video(files, query, embedding_model, domain)
    text_data = [document["text"] for document in context["message"]]
  
    # Parse message history and generation params
    parsed_messages = json.loads(messages)
    generation_params = json.loads(gen_params)
    sampling = not "num_beams" in gen_params

    # Create system prompt only for first interaction
    if len(parsed_messages) == 0 or parsed_messages[0]["role"] != "system":
        system_prompt = create_system_prompt(domain)
        parsed_messages.insert(0, {"role": "system", "content": system_prompt})

    # Create user query prompt for the current interaction
    user_query = parsed_messages.pop()["content"]
    rag_prompt = create_user_prompt(user_query, text_data)

    parsed_messages.append({"role": "user", "content": rag_prompt})
    parsed_messages.append({"role": "assistant", "content": ""})

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