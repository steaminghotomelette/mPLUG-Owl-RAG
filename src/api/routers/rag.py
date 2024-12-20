from datetime import datetime
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Dict
from api.routers import summarizer, chunker
from db.utils import USER_COLLECTION_NAME
from utils.rag_utils import RAGManager

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
    document: UploadFile = File(...),
    metadata: str = Form(...),
    embedding_model: str = Form(...)
) -> Dict:
    """
    Upload a file, extract embeddings, and store it in the database.

    Args:
        document (UploadFile): File uploaded.
        metadata (str): Metadata associated with the file.
        embedding_model (str): Embedding model to use for the file.

    Returns:
        dict: Success message or error details.
    """
    text_embeddings_list = None
    img_embeddings_list = None
    raw_image = None
    text_content = f"{metadata}"
    data = []
    try:
        switch_model(embedding_model)
        # get the file bytes
        file_content = await document.read()
        # Validate file type
        allowed_types = [
            "application/pdf", "image/png", "image/jpeg",
            "video/mp4", "image/gif", "video/x-msvideo"
        ]
        if document.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, detail="Unsupported file type.")
        collection_name = f"{USER_COLLECTION_NAME}_{rag.embedding_model_manager.model_type.value}"

        # Extract embeddings based on file type
        if document.content_type in ["image/png", "image/jpeg", "image/gif"]:
            img_embeddings_list, raw_image = rag.embedding_model_manager.embed_image(
                file_content)
            text_embeddings_list = rag.embedding_model_manager.embed_text(
                text_content)
        else:
            # Chunk and summarize text from PDF
            try:
                chunk_text, chunk_metadata = chunker.split_document(file_content,
                                                                    max_size=200, chunk_overlap=20)
                text_content = summarizer.contextualize_chunks(
                    chunk_text, os.getenv('GEMINI_API_KEY'))
            except Exception as e:
                raise Exception(f"Failed to extract PDF text: {str(e)}")

            for text in text_content:
                text_embeddings_list = rag.embedding_model_manager.embed_text(
                    text)
                data.append({
                    "text": text,
                    "text_embedding": text_embeddings_list,
                    "image_embedding": img_embeddings_list,
                    "image_data": raw_image,
                    "metadata": metadata,
                })

        # Prepare the insertion payload
        if len(data) == 0:
            data.append({
                "text": text_content,
                "text_embedding": text_embeddings_list,
                "image_embedding": img_embeddings_list,
                "image_data": raw_image,
                "metadata": metadata,
            })

        # Insert the data into the database
        response = rag.db.insert(collection_name, data)
        # Return response
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload file: {str(e)}")


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
    files: list[UploadFile] = File(None),
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
    try:
        switch_model(embedding_model)

        file_content = None

        if not files:
            response = rag.search(text=query, image=file_content)
            return response

        response = []
        for file in files:
            # get the file bytes
            file_content = await file.read()
            # Validate file type
            allowed_types = [
                "application/pdf", "image/png", "image/jpeg",
                "video/mp4", "image/gif", "video/x-msvideo"
            ]
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400, detail="Unsupported file type.")

            elif file.content_type in ['video/mp4', 'video/x-msvideo']:
                response.extend(rag.search_video(text=query, video=file_content)['message'])
                    
            else:
                # search
                response.extend(rag.search(text=query, image=file_content)['message'])

        response = rag.deduplicate(response).to_pylist()
        return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'success': True, 'message': response}            

    except Exception as e:
        raise Exception(f"Failed to search: {str(e)}")


def switch_model(embedding_model:str):
    """
    Switch rag manager's embedding model to specified one.
    """
    # Update embedding model
    try:
        rag.update_model(embedding_model)
    except Exception as e:
        raise Exception(f"Fail to switch embedding model: {e}")