import streamlit as st
import requests
from typing import List, Dict
import json
from utils.embed_utils import EmbeddingModel

# --------------------------------------------
# Constants
# --------------------------------------------
PAGE_NAME = "RAG VQA Chat"
API_BASE_URL = "http://127.0.0.1:8000/chat"

# --------------------------------------------
# Utility Functions
# --------------------------------------------
def process_prompt(prompt: str, model: str) -> str:
    """
    Process the user prompt based on the selected model.

    Args:
        prompt (str): The input prompt from the user.
        model (str): The selected LLM model.

    Returns:
        str: The formatted prompt for the model.
    """
    if model == "mPLUG-Owl3-7B":
        return format_for_mplug_owl3(prompt)
    raise ValueError("Invalid model specified!")

def format_for_mplug_owl3(prompt: str) -> str:
    """
    Format the prompt for the mPLUG-Owl3 model with multimedia context.

    Args:
        prompt (str): The input prompt from the user.

    Returns:
        str: The formatted prompt including multimedia markers.
    """
    if st.session_state.get("images_uploader"):
        return f"""{"".join(['<|image|>' for _ in st.session_state.get("images_uploader")])} {prompt}"""
    if st.session_state.get("video_uploader"):
        return f"""<|video|> {prompt}"""
    return prompt

# --------------------------------------------
# API Requests
# --------------------------------------------
def request_mplug_owl3_response(
    message_history: List[Dict[str, str]], image_history: List, video_history: List
) -> Dict:
    """
    Send a request to the backend for a response from the mPLUG-Owl3 model.

    Args:
        message_history (List[Dict[str, str]]): Chat message history.
        image_history (List): List of uploaded images.
        video_history (List): List of uploaded videos.

    Returns:
        Dict: The backend response.
    """
    files = []
    for image in image_history:
        files.append(("files", (image.name, image.getvalue(), image.type)))
    for video in video_history:
        files.append(("files", (video.name, video.getvalue(), video.type)))

    payload = {"message_history": json.dumps(message_history)}

    try:
        response = requests.post(
            f"{API_BASE_URL}/mplug_owl3", data=payload, files=files
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {e}")
        return {"error": str(e)}

# --------------------------------------------
# Callback Functions
# --------------------------------------------
def handle_images_upload() -> None:
    """
    Callback for handling image uploads.
    Disables the video uploader if images are uploaded.
    """
    st.session_state["is_video_uploader_enabled"] = not bool(
        st.session_state.get("images_uploader")
    )

def handle_video_upload() -> None:
    """
    Callback for handling video uploads.
    Disables the image uploader if a video is uploaded.
    """
    st.session_state["is_image_uploader_enabled"] = not bool(
        st.session_state.get("video_uploader")
    )

# --------------------------------------------
# Main Application
# --------------------------------------------
def main() -> None:
    """
    Main function to run the Streamlit app.
    """

    # Page configuration
    st.set_page_config(
        page_title="Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(PAGE_NAME)

    # --------------------------------------------
    # Session State Initialization
    # --------------------------------------------
    for key, default_value in {
        "is_image_uploader_enabled": True,
        "is_video_uploader_enabled": True,
        "message_history": [],
        "image_history": [],
        "video_history": [],
        "messages": [],
        "embedding_model": EmbeddingModel.DEFAULT.value
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --------------------------------------------
    # Sidebar Configuration
    # --------------------------------------------
    st.sidebar.title("LLM Configuration")

    st.sidebar.subheader("Media Upload")

    st.sidebar.file_uploader(
        "Upload Images to LLM",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        disabled=not st.session_state["is_image_uploader_enabled"],
        on_change=handle_images_upload,
        key="images_uploader",
    )

    st.sidebar.file_uploader(
        "Upload Video to LLM",
        type=["gif", "mp4", "avi"],
        disabled=not st.session_state["is_video_uploader_enabled"],
        on_change=handle_video_upload,
        key="video_uploader",
    )

    st.sidebar.subheader("LLM Settings")

    decode_type = st.sidebar.radio(
        "Select decode type",
        ["Beam Search", "Sampling"],
        help="Beam Search: More rational; Sampling: More creative",
    )

    domain = st.sidebar.selectbox(
        "Select domain specialization",
        options=["Medical", "Forensics"],
        help="Choose the domain to tailor the LLM's responses.",
    )

    model = st.sidebar.selectbox(
        "Select model",
        options=["mPLUG-Owl3-7B"],
        help="Choose the LLM to use.",
    )

    # --------------------------------------------
    # Chat Window
    # --------------------------------------------
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your message"):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        formatted_prompt = process_prompt(prompt, model)
        st.session_state["message_history"].append({"role": "user", "content": formatted_prompt})

        if st.session_state.get("images_uploader"):
            st.session_state["image_history"].extend(st.session_state.get("images_uploader"))
        elif st.session_state.get("video_uploader"):
            st.session_state["video_history"].append(st.session_state.get("video_uploader"))

        response = request_mplug_owl3_response(
            st.session_state["message_history"],
            st.session_state["image_history"],
            st.session_state["video_history"],
        )

        llm_response = response.get("response", "Error: No response from backend.")
        st.session_state["messages"].append({"role": "assistant", "content": llm_response})
        st.session_state["message_history"].append({"role": "assistant", "content": llm_response})

        with st.chat_message("assistant"):
            st.markdown(llm_response)

if __name__ == "__main__":
    main()
