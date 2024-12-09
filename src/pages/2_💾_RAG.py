import streamlit as st
from requests import post, get
from typing import List
import db.schemas

# Constants
PAGE_NAME = "RAG Management"
API_BASE_URL = "http://127.0.0.1:8000/rag_documents"

def create_collection(collection_name: str, embedding_model: str) -> dict:
    """
    Create a new collection in the RAG system using the API.

    Args:
        collection_name (str): The name of the collection to create.
        embedding_model (str): The embedding model to associate with the collection.

    Returns:
        dict: Response from the API.
    """
    url = f"{API_BASE_URL}/collections"
    payload = {
        "collection_name": f"{collection_name}_{embedding_model}",
        "embedding_model": embedding_model
    }
    response = post(url, params=payload)
    return response.json()

def upload_to_rag(collection_name: str, file: bytes, file_name: str, metadata: str, content_type: str, embed_model: str) -> dict:
    """
    Upload a file to the RAG system using the API.

    Args:
        collection_name (str): The target collection name.
        file (bytes): The file content in bytes.
        file_name (str): The name of the file.
        metadata (str): Metadata associated with the file.
        content_type (str): MIME type of the file.
        embed_model (str): The embedding model to use.

    Returns:
        dict: Response from the API.
    """
    url = f"{API_BASE_URL}/upload/{collection_name}"
    files = {
        "document": (file_name, file, content_type),

    }
    data = {"metadata": metadata, "embedding_model": embed_model}
    response = post(url, files=files, data=data)
    return response.json()

def fetch_collections() -> List[str]:
    """
    Fetch the list of collections from the RAG API.

    Returns:
        List[str]: A list of collection names.
    """
    url = f"{API_BASE_URL}/collections"
    response = get(url)
    return response.json()

def main() -> None:
    """
    Main function to render the RAG Management app.
    """
    # --------------------------------------------
    # Page Configuration
    # --------------------------------------------
    st.set_page_config(
        page_title="RAG",
        page_icon="💾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # App Title
    st.title(PAGE_NAME)

    # --------------------------------------------
    # Sidebar Configuration
    # --------------------------------------------
    st.sidebar.title("RAG Configuration")

    # Initialize session state for collections if not exists
    if 'collections' not in st.session_state:
        st.session_state.collections = fetch_collections()["collections"]

    # RAG Upload Section
    st.sidebar.subheader("RAG Upload")
    media_file = st.sidebar.file_uploader(
        "Upload Text/Image/Video to RAG System",
        type=["pdf", "png", "jpg", "jpeg", "gif", "mp4", "avi"],
        help="Supported formats: PDF, images (PNG, JPG, JPEG, GIF), videos (MP4, AVI)."
    )

    # Dropdown for collections using session state
    selected_collection = st.sidebar.selectbox(
        "Select Collection",
        options=st.session_state.collections,
        help="Choose the collection to upload the document to."
    )

    # Metadata Input Section
    metadata = st.sidebar.text_input(
        "Enter metadata for the uploaded file",
        help="Provide additional information about the uploaded file."
    )

    # RAG Settings Section
    st.sidebar.subheader("RAG Settings")
    embed_model = st.sidebar.selectbox(
        "Select embedding model",
        options=["CLIP", "BLIP"],
        help="Choose the model used to embed documents."
    )

    # Submit Button for File Upload
    upload_button = st.sidebar.button("Upload")
    if upload_button:
        if media_file:
            try:
                file_content = media_file.read()
                file_name = media_file.name
                content_type = media_file.type

                # Make request to upload file
                response = upload_to_rag(selected_collection, file_content, file_name, metadata, content_type, embed_model)
                st.success("File uploaded successfully!")
                st.json(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload a file.")

    # --------------------------------------------
    # Main Window
    # --------------------------------------------
    st.header("Create New Collection")

    with st.form("create_collection_form"):
        collection_name = st.text_input(
            "Collection Name",
            help="Enter the name of the collection you wish to create."
        )
        create_button = st.form_submit_button("Create Collection")

        if create_button and collection_name and embed_model:
            try:
                # Create the collection
                response = create_collection(collection_name, embed_model)
                
                # Refresh the collections list
                updated_collections = fetch_collections()["collections"]
                st.session_state.collections = updated_collections
                
                st.success("Collection created successfully!")
                st.json(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()