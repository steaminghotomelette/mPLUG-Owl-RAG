from fastapi import APIRouter, UploadFile, File, Form
from utils.rag_manager import RAGManager
from typing import List, Dict

# --------------------------------------------
# Router Initialization
# --------------------------------------------
router = APIRouter(
    prefix="/rag_documents",
    tags=["rag_documents"],
    responses={404: {"description": "Not found"}},
)

# Initialise RAG Manager that contains DBConnection and EmbeddingModelManager
rag = RAGManager()

# --------------------------------------------
# Post Endpoints
# --------------------------------------------
# --------------------------------------------
# Upload to RAG
# --------------------------------------------
@router.post("/upload")
async def upload_document_to_rag(
    files: List[UploadFile] = File(None),
    metadata: List[str] = Form(...),
    embedding_model: str = Form(...),
    domain: str = Form(...),
) -> Dict:
    """
    Upload a file, extract embeddings, and store it in the database.

    Args:
        files (List[UploadFile]): List of files to be uploaded.
        metadata (List[str]): Metadata associated with each uploaded file.
        embedding_model (str): Embedding model to use for the file.
        domain (str): Current domain specified.
    """
    try:
        return await rag.upload(files, metadata, embedding_model, domain)
    except Exception as e:
        raise Exception(f"Failed to upload file: {e}")

# --------------------------------------------
# Reset User RAG collection
# --------------------------------------------
@router.post("/reset_rag")
def reset_user_table():
    """
    Endpoint to reset the user table in the RAG system.
    """
    try:
        return rag.reset_user_table()
    except Exception as e:
        raise Exception(f"Failed to reset user table: {e}")

# --------------------------------------------
# Search RAG
# --------------------------------------------
@router.post("/search")
async def search_image(
    files: List[UploadFile] = File(None),
    query: str = Form(...),
    embedding_model: str = Form(...),
    domain: str = Form(...)
) -> Dict:
    """
    Search top k relevant results from user, multimodal and document tables
    using document and query.

    Args:
        files (list[UploadFile]): List of files uploaded.
        query (str): Query to search RAG.
        embedding_model (str): Embedding model to use for the file.
        domain (str): Current domain specified.
    """
    return await rag.search_image(files, query, embedding_model, domain)

@router.post("/search_video")
async def search_video(
    files: List[UploadFile] = File(None),
    query: str = Form(...),
    embedding_model: str = Form(...),
    domain: str = Form(...)
) -> Dict:
    """
    Search top k relevant results from user, multimodal and document tables
    using document and query.

    Args:
        files (list[UploadFile]): List of files uploaded.
        query (str): Query to search RAG.
        embedding_model (str): Embedding model to use for the file.
        domain (str): Current domain specified.

    Returns:
       Dict containing relevant texts within 'content' key 
    """
    return await rag.search_video(files, query, embedding_model, domain)