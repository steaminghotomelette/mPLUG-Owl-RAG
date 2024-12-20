from io import BytesIO
from typing import Dict, List
from fastapi import UploadFile
import streamlit as st
from requests import post, get, exceptions
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer
import tempfile
from utils.embed_utils import EmbeddingModel
import pandas as pd

# Constants
PAGE_NAME = "RAG Management"
API_BASE_URL = "http://127.0.0.1:8000/rag_documents"


def upload_to_rag(media_file: UploadFile, metadata: str, embed_model: str) -> dict:
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
    files = {
        "document": (media_file.name, media_file.read(), media_file.type),
    }
    data = {"metadata": metadata, "embedding_model": embed_model}
    response = post(url, files=files, data=data)
    return response.json()


def search_rag(files_upload: List, query: str, embed_model: str) -> dict:
    """
    Search RAG system using API.

    Args:
        media_file (UploadFile): The uploaded file.
        query (str): Query to search RAG.
        embed_model (str): The embedding model to use.

    Returns:
        dict: Response from the API.

    """
    try:
        url = f"{API_BASE_URL}/search"
        files = []
        for file in files_upload:
            files.append(("files", (file.name, file.read(), file.type)))

        data = {"query": query, "embedding_model": embed_model}
        response = post(url, files=files, data=data)
        response = response.json()
        return response
    except exceptions.HTTPError as http_err:
        raise Exception(f"HTTP error occurred: {http_err}")
    except Exception as e:
        raise Exception(f"Search RAG failed: {e}")


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
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(result)

    # Drop the unwanted columns
    df = df.drop(columns=["id", "_rowid", "text_embedding", "image_embedding"])
    df['image_data'] = df['image_data'].apply(
        lambda x: f'data:image/jpeg;base64,{x}')

    df.rename(columns={
        'text': 'Text',
        'metadata': 'Metadata',
        '_relevance_score': 'Initial Relevance Score',
        '_weighted_relevance_score': 'Weighted Relevance Score'
    }, inplace=True)

    st.dataframe(df,
                 column_config={
                     "image_data": st.column_config.ImageColumn("Relevant Image")
                 })


def preview_file(media_file: UploadFile):
    """
    Interface for previewing uploaded files at user interface.

    Args:
        media_file (UploadFile): File uploaded via streamlit upload button

    Returns:
        None
    """
    file_name = media_file.name
    content_type = media_file.type
    try:
        # Preview Logic
        left_co, cent_co, last_co = st.columns(3)

        if content_type.startswith("image"):

            raw_image = BytesIO(media_file.read())
            image = Image.open(raw_image)
            raw_image = raw_image.getvalue()

            # If the file is an image
            image = Image.open(media_file)
            original_width, original_height = image.size
            target_width = 300
            target_height = int(
                (target_width / original_width) * original_height)
            resized_image = image.resize((target_width, target_height))
            with cent_co:
                st.image(resized_image,
                         caption=f"Preview of {file_name}",)

        elif content_type == "application/pdf":
            # If the file is a PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(media_file.read())
                temp_pdf_path = temp_pdf.name

            # Use streamlit_pdf_viewer to render PDF
            pdf_viewer(temp_pdf_path, height=600)

        elif content_type.startswith("text") or file_name.endswith(".txt"):
            # If the file is text
            file_content = media_file.read().decode("utf-8")
            st.text_area("Text File Preview", file_content, height=300)

        elif content_type.startswith("video"):
            # If the file is a video
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_name.split('.')[-1]) as temp_video:
                temp_video.write(media_file.read())
                temp_video_path = temp_video.name
            with cent_co:
                st.video(temp_video_path)

        else:
            st.warning("Preview is not supported for this file type.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")


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

    if "embedding_model" not in st.session_state:
        st.session_state['embedding_model'] = EmbeddingModel.DEFAULT.value

    # App Title
    st.title(PAGE_NAME)

    # --------------------------------------------
    # Sidebar Configuration
    # --------------------------------------------
    st.sidebar.title("RAG Configuration")

    # RAG Settings Section
    st.sidebar.subheader("RAG Settings")
    sidebar_text = st.sidebar.empty()
    sidebar_text.text(f"Current embedding model: \
                        {st.session_state.embedding_model}")
    # Get the list of options from the EmbeddingModel enum
    options = [model.value for model in EmbeddingModel]

    # Find the index of the current embedding model in the enum list
    if st.session_state.embedding_model:
        index = options.index(st.session_state.embedding_model)
    else:
        # Default to DEFAULT if no session state is set
        index = options.index(EmbeddingModel.DEFAULT.value)

    embed_model = st.sidebar.selectbox(
        "Select embedding model",
        options=options,
        index=index,
        help="Choose the model used to embed documents."
    )

    if embed_model and embed_model != st.session_state.embedding_model:
        st.sidebar.warning("Switching the embedding model will store results separately. \
                       Files uploaded with a different embedding model will not be accessible."
                           )

    reset_btn = st.sidebar.button("Reset RAG",
                                  help='Resets the RAG documents uploaded.')
    if reset_btn:
        response = reset_rag()
        st.toast(response)

    # --------------------------------------------
    # Main Window
    # --------------------------------------------
    # RAG Upload Section
    with st.expander("RAG Upload", expanded=True):

        media_file = st.file_uploader(
            "Upload Text/Image/Video to RAG System",
            key="user_upload",
            type=["pdf", "png", "jpg", "jpeg", "gif", "mp4", "avi"],
            help="Supported formats: PDF, images (PNG, JPG, JPEG, GIF), videos (MP4, AVI)."
        )

        if media_file:
            st.subheader("File Preview")
            try:
                media_file.seek(0)
                preview_file(media_file)
            except Exception as e:
                raise Exception(f"Error for previewing file: {e}")

        # Metadata Input Section
        metadata = st.text_input(
            "Enter metadata for the uploaded file",
            help="Provide additional information about the uploaded file."
        )

        # Submit Button for File Upload
        upload_button = st.button("Upload")
        if upload_button:
            if media_file:
                try:
                    # Make request to upload file
                    media_file.seek(0)
                    response = upload_to_rag(
                        media_file,
                        metadata,
                        embed_model)
                    if response['success']:
                        st.success(response['message'])
                        st.session_state.embedding_model = embed_model
                        sidebar_text.text(f"Current embedding model: \
                            {st.session_state.embedding_model}")
                    else:
                        st.error(response['message'])
                    st.json(response)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Please upload a file.")

    # --------------------------------------------
    # Search collection
    # --------------------------------------------
    # RAG Search Section
    with st.expander("RAG Search", expanded=True):

        search_media_file = st.file_uploader(
            "Upload Text/Image/Video for Searching",
            accept_multiple_files=True,
            key="search_media",
            type=["pdf", "png", "jpg", "jpeg", "gif", "mp4", "avi"],
            help="Supported formats: PDF, images (PNG, JPG, JPEG, GIF), videos (MP4, AVI)."
        )

        if search_media_file:
            st.subheader("File Preview")
            try:
                for file in search_media_file:
                    file.seek(0)
                    preview_file(file)
            except Exception as e:
                raise Exception(f"Error for previewing file: {e}")

        # Query Input Section
        query = st.text_input(
            "Enter query",
            help="Ask questions about uploaded document."
        )

        # Submit Button for File Upload
        search_button = st.button("Search")

        col1, col2 = st.columns(2)
        with col1:
            # Toggle for displaying response raw json
            response_debug_toggle = st.toggle("Display JSON output",
                                              value=True,
                                              help="Display the latest successful search JSON output")
        with col2:
            # Toggle for visualising search result
            display_search_result_toggle = st.toggle("Visualise result",
                                                     value=True,
                                                     help="Display the latest successful search result")

        if search_button:
            try:
                if search_media_file:
                    for file in search_media_file:
                        file.seek(0)
                try:
                    response = search_rag(
                        search_media_file, query, embed_model)
                except Exception as e:
                    raise Exception(f"Search RAG error: {e}")
                if response['success']:
                    st.session_state.embedding_model = embed_model
                    sidebar_text.text(f"Current embedding model: \
                        {st.session_state.embedding_model}")
                    st.session_state['search_response'] = response

            except Exception as e:
                st.error(f"An error occurred: {e}")

        if 'search_response' in st.session_state and st.session_state.search_response:
            if response_debug_toggle:
                st.json(st.session_state.search_response)

            if display_search_result_toggle:
                display_search_result(
                    st.session_state.search_response['message'])


if __name__ == "__main__":
    main()
