import streamlit as st

# Constants
APP_NAME = "MM-RAG VQA Chat"

def main() -> None:
    """
    Main function to run the Streamlit app.
    """

    # --------------------------------------------
    # Page Configuration
    # --------------------------------------------
    st.set_page_config(
        page_title="Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # App Title
    st.title(APP_NAME)

    # --------------------------------------------
    # Sidebar Configuration
    # --------------------------------------------
    st.sidebar.title("LLM Configuration")

    # Media Upload Section
    st.sidebar.subheader("Media Upload")
    media_file = st.sidebar.file_uploader(
        "Upload Image/Video to LLM",
        type=["png", "jpg", "jpeg", "gif", "mp4", "avi"]
    )

    # LLM Settings Section
    st.sidebar.subheader("LLM Settings")
    decode_type = st.sidebar.radio(
        "Select decode type",
        ["Beam Search", "Sampling"],
        help="Beam Search: More rational; Sampling: More creative"
    )

    domain = st.sidebar.selectbox(
        "Select domain specialization",
        options=["Medical", "Forensics"],
        help="Choose the domain to tailor the LLM's responses."
    )
    st.sidebar.write(f"Selected Domain: {domain}")

    # --------------------------------------------
    # Chat Window
    # --------------------------------------------
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input prompt for the user
    if prompt := st.chat_input("Enter your message"):
        # Append user message to the chat history
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # Display user's message
        with st.chat_message("user"):
            st.markdown(prompt)

        # TODO: Handle API request and add the bot's response to the chat history

if __name__ == "__main__":
    main()
