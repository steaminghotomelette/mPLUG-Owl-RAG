from io import BytesIO
from fastapi import UploadFile
import streamlit as st
from requests import post, get
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer
import tempfile
from db.utils import USER_COLLECTION_NAME
from utils.embed_utils import EmbeddingModel

# Constants
PAGE_NAME = "RAG Management"
API_BASE_URL = "http://127.0.0.1:8000/rag_documents"


def upload_to_rag(collection_name: str, media_file: UploadFile, metadata: str, embed_model: str) -> dict:
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
    url = f"{API_BASE_URL}/upload"
    files = {
        "document": (media_file.name, media_file.read(), media_file.type),
    }
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
        st.subheader("Preview of File to Upload")
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
            type=["pdf", "png", "jpg", "jpeg", "gif", "mp4", "avi"],
            help="Supported formats: PDF, images (PNG, JPG, JPEG, GIF), videos (MP4, AVI)."
        )

        if media_file:
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
                        USER_COLLECTION_NAME,
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


if __name__ == "__main__":
    main()
