from io import BytesIO
from typing import Dict, List
from fastapi import UploadFile
import streamlit as st
from requests import post, get, exceptions
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer
import tempfile
from utils.embed_utils import EmbeddingModel
from utils.rag_utils import search_rag
import pandas as pd

# Constants
PAGE_NAME = "RAG Management"
API_BASE_URL = "http://127.0.0.1:8000/rag_documents"


def upload_to_rag(files_upload: List[UploadFile], metadata: List[str], embed_model: str) -> dict:
    """
    Upload a file to the RAG system using the API.

    Args:
        file (bytes): The file content in bytes.
        file_name (str): The name of the file.
        metadata (str): Metadata associated with the file.
        content_type (str): MIME type of the file.
        embed_model (str): The embedding model to use.

    Returns:
        dict: Response from the API.
    """
    url = f"{API_BASE_URL}/upload"
    files = []
    for file in files_upload:
            files.append(("files", (file.name, file.read(), file.type)))
    data = {"metadata": metadata, "embedding_model": embed_model}
    response = post(url, files=files, data=data)
    return response.json()


def reset_rag():
    """
    Invoke reset of user collection for RAG using API.

    Args:
        None

    Returns:
        dict: Response from API
    """
    url = f"{API_BASE_URL}/reset_rag"
    response = post(url)
    return response.json()


def display_search_result(result: List[Dict]):
    """
    Display RAG search result on interface.

    Args:
        result (List[Dict]): list of dicts containing search result

    Returns:
        None

    """
    if len(result) > 0:
        df = pd.DataFrame(result).drop(columns=["id", "_rowid", "text_embedding", "image_embedding"], errors='ignore')
        if 'image_data' in df.columns:
            df['image_data'] = df['image_data'].apply(lambda x: f'data:image/jpeg;base64,{x}')
            df.rename(columns={
                'text': 'Text',
                'metadata': 'Metadata',
                '_relevance_score': 'Initial Relevance Score',
                '_weighted_relevance_score': 'Weighted Relevance Score'
            }, inplace=True)
            st.dataframe(df, column_config={"image_data": st.column_config.ImageColumn("Relevant Image")})
    else:
        st.warning("No relevant context.")


def preview_file(file: UploadFile):
    """
    Interface for previewing uploaded files at user interface.

    Args:
        file (UploadFile): File uploaded via streamlit upload button

    Returns:
        None
    """
    try:
        content_type = file.type
        file.seek(0)  # Reset file pointer

        if content_type.startswith("image"):
            image = Image.open(BytesIO(file.read()))
            resized_image = image.resize((300, int((300 / image.width) * image.height)))
            st.image(resized_image, caption=f"Preview of {file.name}")

        elif content_type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())
                pdf_viewer(temp_pdf.name, height=600)

        elif content_type.startswith("text") or file.name.endswith(".txt"):
            st.text_area("Text File Preview", file.read().decode("utf-8"), height=300)

        elif content_type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name.split('.')[-1]) as temp_video:
                temp_video.write(file.read())
                st.video(temp_video.name)

        else:
            st.warning("Preview is not supported for this file type.")
    except Exception as e:
        st.error(f"An error occurred while previewing the file: {e}")

def sidebar_configuration():
    """
    Sidebar configuration code for streamlit ui
    """
    st.sidebar.title("RAG Configuration")
    st.sidebar.subheader("RAG Settings")
    sidebar_text = st.sidebar.empty()
    sidebar_text.text(f"Current embedding model: \
                        {st.session_state.embedding_model}")
    
    options = [model.value for model in EmbeddingModel]
    index = options.index(st.session_state.get("embedding_model", EmbeddingModel.DEFAULT.value))
    embed_model = st.sidebar.selectbox("Select embedding model",
                                        options=options, 
                                        index=index,
                                        key="embed_model")
    if embed_model != st.session_state.embedding_model:
        st.sidebar.warning("Switching the embedding model will store results separately.")

    if st.sidebar.button("Reset RAG", help='Resets the RAG documents uploaded.'):
        st.toast(reset_rag())


def main() -> None:
    """
    Main function to render the RAG Management app.
    """
    # --------------------------------------------
    # Page & Sidebar Configuration
    # --------------------------------------------
    st.set_page_config(
        page_title="RAG",
        page_icon="💾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(PAGE_NAME)

    if "embedding_model" not in st.session_state:
        st.session_state['embedding_model'] = EmbeddingModel.DEFAULT.value

    sidebar_configuration()
    
    # --------------------------------------------
    # Main Window
    # --------------------------------------------
    # RAG Upload Section
    with st.expander("RAG Upload", expanded=True):
        files = st.file_uploader("Upload Files",
                                accept_multiple_files=True,
                                type=["pdf", "png", "jpg", "jpeg", "gif"],
                                key="upload")
        if files:
            st.subheader("File Preview")
            for file in files:
                preview_file(file)

            metadata = [st.text_input(f"Metadata for {file.name}") for file in files]

            if st.button("Upload"):
                try:
                    for file in files:
                        file.seek(0)
                    response = upload_to_rag(files, metadata, st.session_state.get("embed_model"))
                    if response.get('success'):
                        st.session_state['embedding_model'] = st.session_state.get("embed_model")
                        st.success(response.get('message', 'Upload successful.'))  
                    else: st.error(response.get('message', 'Upload failed.'))
                    st.json(response)
                except Exception as e:
                    st.error(f"Error during upload: {e}")

    # RAG Search Section
    with st.expander("RAG Search", expanded=True):
        search_media_file = st.file_uploader(
            "Upload Text/Image/Video for Searching",
            accept_multiple_files=True,
            key="search_media",
            type=["pdf", "png", "jpg", "jpeg", "gif"],
            help="Supported formats: PDF, images (PNG, JPG, JPEG, GIF)"
        )

        if search_media_file:
            st.subheader("File Preview")
            for file in search_media_file:
                try:
                    file.seek(0)
                    preview_file(file)
                except Exception as e:
                    st.error(f"Error previewing file {file.name}: {e}")

        # Query Input Section
        query = st.text_input(
            "Enter query",
            help="Ask questions about uploaded document."
        )

        # Submit Button for File Upload
        if st.button("Search"):
            try:
                if search_media_file:
                    for file in search_media_file:
                        file.seek(0)
                response = search_rag(search_media_file, query, st.session_state.get("embed_model"))
                if response.get('success'):
                    st.session_state['embedding_model'] = st.session_state.get("embed_model")
                    st.session_state['search_response'] = response
                    st.success("Search completed successfully.")
                else:
                    st.error(response.get('message', 'Search failed.'))
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

        col1, col2 = st.columns(2)
        with col1:
            response_debug_toggle = st.checkbox(
                "Display JSON output",
                value=True,
                help="Display the latest successful search JSON output"
            )
        with col2:
            display_search_result_toggle = st.checkbox(
                "Visualise result",
                value=True,
                help="Display the latest successful search result"
            )

        if st.session_state.get('search_response'):
            if response_debug_toggle:
                st.json(st.session_state['search_response'])

            if display_search_result_toggle:
                display_search_result(st.session_state['search_response'].get('message', []))


if __name__ == "__main__":
    main()
