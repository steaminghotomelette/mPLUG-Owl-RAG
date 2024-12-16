from io import BytesIO
from typing import List, Dict, Tuple
import pymupdf
from langchain.text_splitter import CharacterTextSplitter


def split_document(document: BytesIO, max_size: int = 100, chunk_overlap: int = 20) -> Tuple[List[str], List[Dict]]:
    """Splits a PDF document into chunks of text.

    Args:
        document (BytesIO): PDF document as a byte stream.
        max_size (int, optional): Maximum chunk size in characters. Default is 100.
        chunk_overlap (int, optional): Number of overlapping characters between chunks. Default is 20.

    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing two lists:
            - List of text chunks.
            - List of metadata dictionaries, each containing the page number for the corresponding chunk.
    """
    try:
        chunks = []
        splitter = CharacterTextSplitter(
            separator="\n",
            keep_separator=True,
            chunk_size=max_size,
            chunk_overlap=chunk_overlap,
        )

        doc = pymupdf.open(stream=document, filetype='pdf')
        # Initialize variables
        chunks = []
        current_text = ""

        # Loop through pages
        for page in doc:
            # Extract all text from the page in one go
            text = page.get_text("text")

            # If the extracted text exceeds max_size, split into smaller chunks
            if len(text) > max_size:
                split_chunks = splitter.split_text(text)
                for chunk in split_chunks:
                    chunks.append(
                        {"text": chunk, "metadata": {"page": page.number}})
            else:
                # If adding the text exceeds max_size, save the current chunk and start a new one
                if len(current_text + text) >= max_size:
                    if current_text:
                        chunks.append(
                            {"text": current_text, "metadata": {"page": page.number}})
                    current_text = text
                else:
                    # Accumulate text for the current chunk
                    current_text += text

        # Finalize the last chunk
        if current_text.strip():
            chunks.append(
                {"text": current_text, "metadata": {"page": doc.page_count}})

        # Close the document
        doc.close()

        # Filter out chunks with only whitespace
        chunks = [chunk for chunk in chunks if chunk["text"].strip()]

        # Separate output into text and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadata = [chunk["metadata"] for chunk in chunks]

        return texts, metadata
    except Exception as e:
        raise Exception(f"Split document failed: {e}")
