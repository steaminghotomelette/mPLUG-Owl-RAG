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
    chunks = []
    splitter = CharacterTextSplitter(
        separator="\n",
        keep_separator=True,
        chunk_size=max_size,
        chunk_overlap=chunk_overlap,
    )

    doc = pymupdf.open(stream=BytesIO(document))
    current_text = ""

    for page in doc:
        # Extract text blocks from page, and combine them
        blocks = page.get_text_blocks()
        text = "".join([b[4] for b in blocks]) 

        # If too large, split into chunks
        if len(text) > max_size:
            chunks.append(
                {"text": current_text, "metadata": {"page": page.number}}
            )
            chunks = splitter.split_text(text)
            for chunk in chunks:
                chunks.append(
                    {"text": chunk, "metadata": {"page": page.number}}
                )
            current_text = ""

        # If adding new text exceeds max_size, finalize current chunk and start a new one
        elif len(current_text + text) >= max_size:
            if current_text != "":
                chunks.append(
                    {"text": current_text, "metadata": {"page": page.number}}  # Add current chunk
                )
            current_text = text

        # Otherwise, add more text for current chunk
        else:
            current_text += text

    # Remove whitespace chunks
    chunks = [
        chunk for chunk in chunks if chunk["text"].strip().replace(" ", "")
    ]
    
    # Separate output
    texts = [chunk["text"] for chunk in chunks]
    metadata = [chunk["metadata"] for chunk in chunks]

    return texts, metadata