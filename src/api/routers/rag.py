from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from io import BytesIO
from typing import Dict
from PyPDF2 import PdfReader
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
    text_content = f"\nContext: {metadata}"
    try:
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
        # Update embedding model
        try:
            rag.update_model(embedding_model)
        except Exception as e:
            raise Exception(f"Fail to switch embedding model: {e}")
        collection_name = f"{USER_COLLECTION_NAME}_{
            rag.embedding_model_manager.model_type.value}"

        # Extract embeddings based on file type
        if document.content_type in ["image/png", "image/jpeg", "image/gif"]:
            img_embeddings_list, raw_image = rag.embedding_model_manager.embed_image(
                file_content)
            text_embeddings_list = rag.embedding_model_manager.embed_text(
                text_content)
        else:
            # Extract text from PDF
            try:
                pdf_reader = PdfReader(BytesIO(file_content))
                text_content = " ".join([page.extract_text()
                                        for page in pdf_reader.pages]) + text_content
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Failed to extract PDF text: {str(e)}")

            text_embeddings_list = rag.embedding_model_manager.embed_text(
                text_content)

        # Prepare the insertion payload
        data = {
            "text": text_content,
            "text_embedding": text_embeddings_list,
            "image_embedding": img_embeddings_list,
            "image_data": raw_image,
            "metadata": metadata,
        }

        # Insert the data into the database
        response = rag.db.insert(collection_name, [data])
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
