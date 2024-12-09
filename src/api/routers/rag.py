from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from db.connection import DBConnection
from transformers import CLIPProcessor, CLIPModel, Blip2Model, AutoTokenizer, AutoProcessor
from PIL import Image
from io import BytesIO
from typing import Dict, List
from PyPDF2 import PdfReader

# Initialize API router
router = APIRouter(
    prefix="/rag_documents",
    tags=["rag_documents"],
    responses={404: {"description": "Not found"}},
)

# Initialize database connection
db = DBConnection()

# Load embedding models (using CLIP)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

blip_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

# --------------------------------------------
# Post Endpoints
# --------------------------------------------

@router.post("/collections")
async def create_collection(collection_name: str, embedding_model: str) -> Dict:
    """
    Create a collection in the database.

    Args:
        collection_name (str): Name of the collection to create.
        embedding_model (str): Embedding model to associate with the collection.

    Returns:
        dict: Success message or error details.
    """
    try:
        db.create_collection(collection_name, embedding_model=embedding_model)
        return {"message": "Collection created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")


@router.post("/upload/{collection_name}")
async def upload_document_to_rag(
    collection_name: str,
    document: UploadFile = File(...),
    metadata: str = Form(...),
    embedding_model: str = Form(...)
) -> Dict:
    """
    Upload a file, extract embeddings, and store it in the database.

    Args:
        collection_name (str): Name of the collection to upload the file to.
        document (UploadFile): File to upload.
        metadata (str): Metadata associated with the file.
        embedding_model (str): Embedding model to use for the file.

    Returns:
        dict: Success message or error details.
    """
    match embedding_model:
        case "CLIP":
            processor = clip_processor
            model = clip_model
        case "BLIP":
            processor = blip_processor
            model = blip_model
    
    try:
        # Read the file content
        file_content = await document.read()

        # Validate file type
        allowed_types = [
            "application/pdf", "image/png", "image/jpeg",
            "video/mp4", "image/gif", "video/x-msvideo"
        ]
        if document.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        
        # Extract embeddings based on file type
        embeddings = None
        if document.content_type in ["image/png", "image/jpeg", "image/gif"]:
            # Convert file content to a PIL image
            try:
                image = Image.open(BytesIO(file_content))

            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image format.")
            
            inputs = processor(images=image, return_tensors="pt", padding=True)
            embeddings = model.get_image_features(**inputs).detach().numpy()
        else:
            # Extract text from PDF
            try:
                pdf_reader = PdfReader(BytesIO(file_content))
                text_content = " ".join([page.extract_text() for page in pdf_reader.pages])
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to extract PDF text: {str(e)}")
            
            # Create embeddings from extracted text
            inputs = processor(text=[text_content], return_tensors="pt", padding=True)
            embeddings = model.get_text_features(**inputs)[0].detach().numpy()
        
        # Convert numpy array to a list of floats to match float_vector requirement
        embeddings_list = embeddings.flatten().tolist()
        
        # Prepare the insertion payload
        data = {
            "embedding": embeddings_list,  # Convert to list for serialization
            "metadata": metadata,
            "storage_link": "placeholder",  # TODO: Add actual storage link if applicable
        }
        
        # Insert the data into the database
        db.insert(collection_name, [data])
        
        # Return success response
        return {"message": "File uploaded successfully."}
   
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

# --------------------------------------------
# Get Endpoints
# --------------------------------------------

@router.get("/collections")
async def list_collections() -> Dict[str, List[str]]:
    """
    Retrieve all collections in the database.

    Returns:
        dict: A dictionary containing a list of collection names.
    """
    try:
        collections = db.list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch collections: {str(e)}")
