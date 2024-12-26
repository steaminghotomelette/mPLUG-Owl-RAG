import streamlit as st
import requests
import json
from typing import List, Dict, Any

# --------------------------------------------
# Constants
# --------------------------------------------
PAGE_NAME = "RAG VQA Chat"
API_BASE_URL = "http://127.0.0.1:8000/mplug_owl3"
STREAMING = True

# --------------------------------------------
# Utility Functions
# --------------------------------------------
def process_prompt(prompt: str) -> str:
    # Check for images in the session state and format them accordingly
    if st.session_state.get("image_uploader"):
        image_markers = "".join([f"<|image|>" for _ in st.session_state["image_uploader"]])
        return f"{image_markers}\n{prompt}"

    # Check for video in the session state and format it accordingly
    elif st.session_state.get("video_uploader"):
        return f"<|video|>\n{prompt}"
    
    else:
        return prompt

# --------------------------------------------
# API Requests
# --------------------------------------------
def request_model_response(
        session_type: str,
        files: List,
        formatted_messages: List[Dict[str, str]],
        gen_params: Dict[str, Any],
        prompt: str,
        streaming: bool = STREAMING
) -> Dict[str, str]:
    
    # Process files
    files = [("files", (file.name, file.getvalue(), file.type)) for file in files]

    endpoint = "/image_chat" if session_type == "image_session" else "/video_chat"
    
    # Prepare payload
    payload = {
        "messages": json.dumps(formatted_messages),
        "gen_params": json.dumps(gen_params),
        "query": prompt,
        "streaming": streaming
    }

    # Make request
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            data=payload,
            files=files,
            stream=streaming
        )
        if streaming:
            response.raise_for_status()
            return response.iter_lines(decode_unicode=True)
        else:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {e}")
        return {"error": str(e)}
    
# --------------------------------------------
# Callback Functions
# --------------------------------------------
def handle_image_upload() -> None:
    if st.session_state["session_type"] == None:
        st.session_state["is_video_uploader_enabled"] = not bool(
            st.session_state.get("image_uploader")
        )

def handle_video_upload() -> None:
    if st.session_state["session_type"] == None:
        st.session_state["is_image_uploader_enabled"] = not bool(
            st.session_state.get("video_uploader")
        )
    
def handle_prompt_submission() -> None:
    if st.session_state["session_type"] == None:

        # Locked into image session
        if st.session_state.get("image_uploader"):
            st.session_state["session_type"] = "image_session"
            st.session_state["is_video_uploader_enabled"] = False
        
        # Locked into video session
        elif st.session_state.get("video_uploader"):
            st.session_state["session_type"] = "video_session"
            st.session_state["is_image_uploader_enabled"] = False

# --------------------------------------------
# Main Application
# --------------------------------------------
def main() -> None:

    # Page config
    st.set_page_config(
        page_title="Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(PAGE_NAME)

    # --------------------------------------------
    # Session State Initialization
    # --------------------------------------------
    for key, default_value in {
        "formatted_messages": [],
        "uploaded_files": [],
        "messages": [],
        "is_image_uploader_enabled": True,
        "is_video_uploader_enabled": True,
        "session_type": None,
        "gen_params": {}
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # --------------------------------------------
    # Sidebar Configuration
    # --------------------------------------------
    st.sidebar.title("LLM Configuration")

    # File uploaders
    st.sidebar.subheader("Media Upload")

    st.sidebar.file_uploader(
        "Upload Images to LLM",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="image_uploader",
        disabled=not st.session_state["is_image_uploader_enabled"],
        on_change=handle_image_upload
    )

    st.sidebar.file_uploader(
        "Upload Video to LLM",
        type=["gif", "mp4", "avi"],
        key="video_uploader",
        disabled=not st.session_state["is_video_uploader_enabled"],
        on_change=handle_video_upload
    )

    # LLM Config
    st.sidebar.subheader("LLM Settings")

    decode_type = st.sidebar.radio(
        "Select decode type",
        ["Sampling", "Beam Search"],
        help=" Sampling: More creative | Beam Search: More rational",
    )

    if decode_type == "Beam Search":
        STREAMING = False
        st.session_state["gen_params"] = {}
        st.session_state["gen_params"]["num_beams"] = st.sidebar.selectbox(
            "Number of beams",
            options=[1, 2, 3, 4, 5, 8, 10],
            index=2,
            help="Higher values consider more alternative sequences but increase computation time.",
        )

    else: # Sampling
        STREAMING = True
        st.session_state["gen_params"] = {}
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.session_state["gen_params"]["temperature"] = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Higher values make output more random, lower values more deterministic.",
            )
            
            st.session_state["gen_params"]["top_p"] = st.slider(
                "Top P",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.1,
                help="Nucleus sampling: cumulative probability cutoff for token selection.",
            )
        
        with col2:
            st.session_state["gen_params"]["top_k"] = st.slider(
                "Top K",
                min_value=1,
                max_value=100,
                value=100,
                step=1,
                help="Limits the number of tokens considered for each step of text generation.",
            )

    domain = st.sidebar.selectbox(
        "Select domain specialization",
        options=["Medical", "Forensics"],
        help="Choose the domain to tailor the LLM's responses.",
    )

    # --------------------------------------------
    # Chat Window
    # --------------------------------------------
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input(
        "Enter your message", 
        on_submit=handle_prompt_submission,
        disabled=not (st.session_state["image_uploader"] or st.session_state["video_uploader"] or st.session_state["messages"])
        ):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)
        
        formatted_prompt = process_prompt(prompt)
        st.session_state["formatted_messages"].append({"role": "user", "content": formatted_prompt})

        if st.session_state.get("image_uploader"):
            st.session_state["uploaded_files"].extend(st.session_state.get("image_uploader"))
        elif st.session_state.get("video_uploader"):
            st.session_state["uploaded_files"].append(st.session_state.get("video_uploader"))
        
        with st.chat_message("assistant"):

            # Request model response
            response = request_model_response(
                st.session_state["session_type"],
                st.session_state["uploaded_files"],
                st.session_state["formatted_messages"],
                st.session_state["gen_params"],
                prompt,
                streaming=STREAMING
            )
            llm_response = ""

            if STREAMING:
                try:
                    llm_response = st.write_stream(response)
                except Exception as e:
                    st.error(f"Error during streaming: {str(e)}")
            else:
                llm_response = response.get("response", "Error: No response from backend.")
                st.write(llm_response)
            
            st.session_state["messages"].append({"role": "assistant", "content": llm_response})
            st.session_state["formatted_messages"].append({"role": "assistant", "content": llm_response})
        
if __name__ == "__main__":
    main()