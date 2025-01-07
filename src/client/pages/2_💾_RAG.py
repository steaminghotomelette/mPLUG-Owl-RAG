from io import BytesIO
from typing import Dict, List
from fastapi import UploadFile
from requests import post
from PIL import Image
from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import tempfile
import pandas as pd

# Constants
PAGE_NAME = "RAG Management"
API_BASE_URL = "http://127.0.0.1:8080/rag_documents"

# --------------------------------------------
# API Calls
# --------------------------------------------
def upload_to_rag(files_upload: List[UploadFile], metadata: List[str], embed_model: str, domain: str) -> dict:
    """
    Upload a file to the RAG system using the API.

    Args:
        file (bytes): The file content in bytes.
        file_name (str): The name of the file.
        metadata (str): Metadata associated with the file.
        content_type (str): MIME type of the file.
        embed_model (str): The embedding model to use.
        domain (str): The domain to use.
    """
    url = f"{API_BASE_URL}/upload"
    files = []
    for file in files_upload:
            files.append(("files", (file.name, file.read(), file.type)))
    data = {"metadata": metadata, "embedding_model": embed_model, "domain": domain}
    response = post(url, files=files, data=data)
    return response.json()

def search_rag_image(files_upload: List, query: str, embed_model: str, domain: str) -> dict:
    """
    Search RAG system using API.

    Args:
        media_file (UploadFile): The uploaded file.
        query (str): Query to search RAG.
        embed_model (str): The embedding model to use.
        domain (str): The domain to use.

    Returns:
        dict: Response from the API.

    """
    try:
        url = f"{API_BASE_URL}/search"
        files = []
        for file in files_upload:
            files.append(("files", (file.name, file.read(), file.type)))

        data = {"query": query, "embedding_model": embed_model, "domain": domain}
        response = post(url, files=files, data=data)
        return response.json()
    except Exception as e:
        raise Exception(f"Search RAG failed: {e}")

def search_rag_video(files_upload: List, query: str, embed_model: str, domain: str) -> dict:
    """
    Search RAG system using API.

    Args:
        media_file (UploadFile): The uploaded file.
        query (str): Query to search RAG.
        embed_model (str): The embedding model to use.
        domain (str): The domain to use.

    Returns:
        dict: Response from the API.

    """
    try:
        url = f"{API_BASE_URL}/search_video"
        files = []
        for file in files_upload:
            files.append(("files", (file.name, file.read(), file.type)))

        data = {"query": query, "embedding_model": embed_model, "domain": domain}
        response = post(url, files=files, data=data)
        return response.json()
    except Exception as e:
        raise Exception(f"Search RAG failed: {e}")

def reset_rag(domain: str):
    """
    Invoke reset of user collection for RAG using API.
    """
    url = f"{API_BASE_URL}/reset_rag"
    data = {"domain": domain}
    response = post(url, data=data)
    return response.json()

# --------------------------------------------
# Display functions
# --------------------------------------------
def display_search_result(result: List[Dict]):
    """
    Display RAG search result on interface.

    Args:
        result (List[Dict]): list of dicts containing search result
    """
    if len(result) > 0:
        cols = ["text", "image_data", "title", "_relevance_score"]
        df = pd.DataFrame(result)
        disp_cols = [col for col in cols if col in df.columns]
        df = df[disp_cols]
        df.rename(columns={
                'text': 'Text',
                'title': 'Title',
                '_relevance_score': 'Relevance Score',
            }, inplace=True)
        if 'image_data' in df.columns:
            df['image_data'] = df['image_data'].apply(lambda x: f'data:image/jpeg;base64,{x}')
            st.dataframe(df, column_config={"image_data": st.column_config.ImageColumn("Relevant Image")})
        else:
            st.dataframe(df)
    else:
        st.warning("No relevant context.")

def preview_file(file: UploadFile):
    """
    Interface for previewing uploaded files at user interface.

    Args:
        file (UploadFile): File uploaded via streamlit upload button
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

# --------------------------------------------
# Sidebar Window
# --------------------------------------------
def sidebar_configuration():
    """
    Sidebar configuration code for streamlit ui
    """
    st.sidebar.title("RAG Configuration")
    st.sidebar.subheader("RAG Settings")
    st.sidebar.text(f"Current domain: \n{st.session_state.domain}")
    sidebar_text = st.sidebar.empty()
    sidebar_text.text(f"Current embedding model: \n{st.session_state.embedding_model}")
    
    options = ["BLIP"]
    index = options.index(st.session_state.get("embedding_model", "BLIP"))
    embed_model = st.sidebar.selectbox("Select embedding model",
                                        options=options, 
                                        index=index,
                                        key="embed_model")
    if embed_model != st.session_state.embedding_model:
        st.sidebar.warning("Switching the embedding model will store results separately.")

    if st.sidebar.button("Reset RAG", help='Resets the RAG documents uploaded.'):
        st.toast(reset_rag(st.session_state.get("domain")))

    

# --------------------------------------------
# Main Application
# --------------------------------------------
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
        st.session_state['embedding_model'] = "BLIP"
    
    if "domain" not in st.session_state:
        st.session_state['domain'] = "Medical"

    sidebar_configuration()
    
    # --------------------------------------------
    # RAG Upload
    # --------------------------------------------
    with st.expander("RAG Upload", expanded=True):
        files = st.file_uploader("Upload Files",
                                accept_multiple_files=True,
                                type=["pdf", "png", "jpg", "jpeg"],
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
                    response = upload_to_rag(files, metadata, st.session_state.get("embed_model"), st.session_state.get("domain"))
                    if response.get('success'):
                        st.session_state['embedding_model'] = st.session_state.get("embed_model")
                        st.session_state['domain'] = st.session_state.get("domain")
                        st.success(response.get('message', 'Upload successful.'))  
                    else: st.error(response.get('message', 'Upload failed.'))
                    st.json(response)
                except Exception as e:
                    st.error(f"Error during upload: {e}")

    # --------------------------------------------
    # Search Image
    # --------------------------------------------
    with st.expander("RAG Image Search", expanded=True):
        search_media_file = st.file_uploader(
            "Upload Image for Searching",
            accept_multiple_files=True,
            key="search_media",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PDF, images (PNG, JPG, JPEG)"
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
                response = search_rag_image(search_media_file, query, st.session_state.get("embed_model"), st.session_state.get("domain"))
                if response.get('success'):
                    st.session_state['embedding_model'] = st.session_state.get("embed_model")
                    st.session_state['search_response'] = response
                    st.session_state['domain'] = st.session_state.get("domain")
                    st.success("Search completed successfully.")
                else:
                    st.error(response.get('content', 'Search failed.'))
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

        # --------------------------------------------
        # Display Search Result
        # --------------------------------------------
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
                display_search_result(st.session_state['search_response'].get('content', []))

    # --------------------------------------------
    # RAG Video Search
    # --------------------------------------------
    with st.expander("RAG Video Search", expanded=True):
        search_media_file = st.file_uploader(
            "Upload Video for Searching",
            accept_multiple_files=True,
            type=["mp4", "avi"],
            help="Supported formats: PDF, images (PNG, JPG, JPEG)"
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
            help="Ask questions about uploaded document.",
            key="query2"
        )

        # Submit Button for File Upload
        if st.button("Search", key="search2"):
            try:
                if search_media_file:
                    for file in search_media_file:
                        file.seek(0)
                response = search_rag_video(search_media_file, query, st.session_state.get("embed_model"), st.session_state.get("domain"))
                if response.get('success'):
                    st.session_state['embedding_model'] = st.session_state.get("embed_model")
                    st.session_state['search_vid_response'] = response
                    st.session_state['domain'] = st.session_state.get("domain")
                    st.success("Search completed successfully.")
                else:
                    st.error(response.get('content', 'Search failed.'))
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

        # --------------------------------------------
        # Display Search Result
        # --------------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            response_debug_toggle = st.checkbox(
                "Display JSON output",
                value=True,
                help="Display the latest successful search JSON output",
                key="debug2"
            )
        with col2:
            display_search_result_toggle = st.checkbox(
                "Visualise result",
                value=True,
                help="Display the latest successful search result",
                key="display2"
            )

        if st.session_state.get('search_vid_response'):
            if response_debug_toggle:
                st.json(st.session_state['search_vid_response'])

            if display_search_result_toggle:
                display_search_result(st.session_state['search_vid_response'].get('content', []))

if __name__ == "__main__":
    main()
