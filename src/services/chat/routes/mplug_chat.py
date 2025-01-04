import json
import os
import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from utils.mplug_manager import MplugOwl3ModelManager
from utils.prompt_utils import filter_stream, create_system_prompt, create_image_based_prompt, create_video_based_prompt
from typing import List, Dict, Union, Any

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
MPLUG_MEDICAL_MODEL_PATH   = os.getenv("MPLUG_MEDICAL_MODEL_PATH")
MPLUG_FORENSICS_MODEL_PATH = os.getenv("MPLUG_FORENSICS_MODEL_PATH")
ATTN_IMPLEMENTATION        = os.getenv("ATTN_IMPLEMENTATION")
RAG_API_BASE_URL           = "http://127.0.0.1:8080/rag_documents"

# Mplug model manager
MPLUG_MEDICAL_MODEL   = MplugOwl3ModelManager(MPLUG_MEDICAL_MODEL_PATH, ATTN_IMPLEMENTATION)
MPLUG_FORENSICS_MODEL = MplugOwl3ModelManager(MPLUG_FORENSICS_MODEL_PATH, ATTN_IMPLEMENTATION)
DOMAIN_MODEL_MAP      = {"Medical": MPLUG_MEDICAL_MODEL, "Forensics": MPLUG_FORENSICS_MODEL}

# --------------------------------------------
# Request Functions
# --------------------------------------------
async def search_rag_image(files_upload: List[UploadFile], query: str, embed_model: str, domain: str) -> Dict[str, Any]:
    try:
        url = f"{RAG_API_BASE_URL}/search"
        
        # Prepare files for async upload
        files = []
        for file in files_upload:
            content = await file.read()
            files.append(("files", (file.filename, content, file.content_type)))

        data = {"query": query, "embedding_model": embed_model, "domain": domain}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, data=data, timeout=None)
            return response.json()
            
    except Exception as e:
        raise Exception(f"Search RAG failed: {e}")

async def search_rag_video(files_upload: List[UploadFile], query: str, embed_model: str, domain: str) -> Dict[str, Any]:
    try:
        url = f"{RAG_API_BASE_URL}/search_video"
        
        # Prepare files for async upload
        files = []
        for file in files_upload:
            content = await file.read()
            files.append(("files", (file.filename, content, file.content_type)))

        data = {"query": query, "embedding_model": embed_model, "domain": domain}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, data=data, timeout=None)
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
    prompt_data = create_image_based_prompt(user_query, context)
    rag_prompt = prompt_data[1]

    # Update the images list
    temp = files.pop()
    files.extend(prompt_data[0])
    files.append(temp)

    parsed_messages.append({"role": "user", "content": rag_prompt})
    parsed_messages.append({"role": "assistant", "content": ""})

    # Generate response
    mplug_manager = DOMAIN_MODEL_MAP["domain"]
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
    rag_prompt = create_video_based_prompt(user_query, context)

    parsed_messages.append({"role": "user", "content": rag_prompt})
    parsed_messages.append({"role": "assistant", "content": ""})

    # Generate response
    mplug_manager = DOMAIN_MODEL_MAP["domain"]
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