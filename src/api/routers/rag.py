from datetime import datetime
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Dict, List
from api.routers import summarizer, chunker
from db.utils import USER_COLLECTION_NAME
from utils.rag_utils import RAGManager, switch_model, rag_search, UPLOAD_ALLOWED_TYPES

# Initialize API router
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
@router.post("/upload")
async def upload_document_to_rag(
    files: List[UploadFile] = File(None),
    metadata: List[str] = Form(...),
    embedding_model: str = Form(...)
) -> Dict:
    """
    Upload a file, extract embeddings, and store it in the database.

    Args:
        files (List[UploadFile]): List of files to be uploaded.
        metadata (List[str]): Metadata associated with each uploaded file.
        embedding_model (str): Embedding model to use for the file.

    Returns:
        dict: Success message or error details.
    """
    data = []
    try:
        switch_model(rag, embedding_model)
        collection_name = f"{USER_COLLECTION_NAME}_{rag.embedding_model_manager.model_type.value}"

        for i, file in enumerate(files):
            if file.content_type not in UPLOAD_ALLOWED_TYPES:
                raise HTTPException(
                    status_code=400, detail="Unsupported file type.")
            
            # get the file bytes
            file_content = await file.read()

            # Extract embeddings based on file type
            if file.content_type in ["image/png", "image/jpeg", "image/gif"]:
                img_embeddings_list, raw_image = rag.embedding_model_manager.embed_image(file_content)
                text_embeddings_list = rag.embedding_model_manager.embed_text(metadata[i])
                data.append({
                    "text": metadata[i],
                    "text_embedding": text_embeddings_list,
                    "image_embedding": img_embeddings_list,
                    "image_data": raw_image,
                    "metadata": metadata[i],
                })

            else:
                # Chunk and summarize text from PDF
                try:
                    chunk_text, chunk_metadata = chunker.split_document(file_content, max_size=200, chunk_overlap=20)
                    text_content = summarizer.contextualize_chunks(chunk_text, os.getenv('GEMINI_API_KEY'))
                except Exception as e:
                    raise Exception(f"Failed to extract PDF text: {str(e)}")

                for text in text_content:
                    text_embeddings_list = rag.embedding_model_manager.embed_text(text)
                    data.append({
                        "text": text,
                        "text_embedding": text_embeddings_list,
                        "image_embedding": None,
                        "image_data": None,
                        "metadata": metadata[i],
                    })

        # Insert the data into the database
        response = rag.db.insert(collection_name, data)
        # Return response
        return response

    except Exception as e:
        raise Exception(f"Failed to upload file: {e}")


@router.post("/reset_rag")
def reset_user_table():
    """
    Endpoint to reset the user table in the RAG system.

    This endpoint is used to clear all data in the user table of the RAG system.

    Returns:
        None
    """
    try:
        return rag.reset_user_table()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset user table: {str(e)}")


@router.post("/search")
async def search_rag(
    files: List[UploadFile] = File(None),
    query: str = Form(...),
    embedding_model: str = Form(...)
) -> Dict:
    """
    Search top k relevant results from user, multimodal and document tables
    using document and query.

    Args:
        files (list[UploadFile]): List of files uploaded.
        query (str): Query to search RAG.
        embedding_model (str): Embedding model to use for the file.

    Returns:
        dict: Success message or error details.
    """
    return await rag_search(rag, files, query, embedding_model)