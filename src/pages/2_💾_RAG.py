import streamlit as st

# Constants
PAGE_NAME = "RAG Management"

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
    st.sidebar.title("RAG Upload")

    # RAG Upload Section
    media_file = st.sidebar.file_uploader(
        "Upload Text/Image/Video to RAG System",
        type=["pdf", "png", "jpg", "jpeg", "gif", "mp4", "avi"],
        help="Supported formats: PDF, images (PNG, JPG, JPEG, GIF), videos (MP4, AVI)."
    )

    # Submit Button TODO
    if st.sidebar.button("Submit"):
        pass

if __name__ == "__main__":
    main()
