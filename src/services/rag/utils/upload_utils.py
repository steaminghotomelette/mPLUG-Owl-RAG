import requests
import json
import math
from io import BytesIO
from typing import List, Dict, Tuple
import pymupdf
from langchain.text_splitter import CharacterTextSplitter

def chunk_document(document: BytesIO, max_size: int = 100, chunk_overlap: int = 20) -> Tuple[List[str], List[Dict]]:
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

def contextualize_chunks(chunks, api_key, batch_size=5):
    """
    Uses the Gemini API to provide context for each chunk of text in larger batches, expecting JSON output.

    Args:
        chunks (List[str]): Text chunks (when combined, comprise the entire document).
        api_key (str): API key for the Gemini API.
        batch_size (int): Number of chunks to process per API call (batch size).

    Returns:
        List[str]: List of text chunks with context.
    """
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        document = " ".join(chunks)
        new_chunks = []

        # Split the chunks into batches
        num_batches = math.ceil(len(chunks) / batch_size)
        
        for i in range(num_batches):
            batch_chunks = chunks[i * batch_size : (i + 1) * batch_size]
            prompt = f"""
            <document>
            {document}
            </document>
            Here are the chunks we want to situate within the whole document:
            <chunks>
            {' '.join(batch_chunks)}
            </chunks>
            Please give a short succinct context to situate each of these chunks within the overall document for the purposes of improving search retrieval in a medical VQA context. 
            Return your response in the following JSON format:
            {{
                "chunks": [
                    {{
                        "chunk": "<chunk1>",
                        "context": "<context1>"
                    }},
                    {{
                        "chunk": "<chunk2>",
                        "context": "<context2>"
                    }},
                    ...
                ]
            }}
            """

            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            }

            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(f"{url}?key={api_key}", headers=headers, data=json.dumps(data))

            if response.status_code == 200:
                response_data = response.json()

                # Parse the JSON content
                chunk_contexts = response_data['candidates'][0]['content']['parts'][0]['text']
                chunk_contexts_json = json.loads(chunk_contexts)

                for item in chunk_contexts_json['chunks']:
                    context = item.get('context', '')
                    chunk = item.get('chunk', '')

                    if context:
                        new_chunk = f"{context} {chunk}"
                        new_chunks.append(new_chunk)
                    else:
                        new_chunks.append(chunk)

            else: # In case of failure, add chunks without context
                print(f"API call failed with status code {response.status_code}")
                print((f"Reponse: {response.text}"))
                new_chunks.extend(batch_chunks)

        return new_chunks
    except Exception as e:
        raise Exception(f"Contextualize chunks failed: {e}")

def summarize_text(chunk: str, api_key: str) -> str:
    """
    Uses the Gemini API to provide context for each chunk of text.

    Args:
        chunks (List[str]): Text chunks (when combined, comprise the entire document).
        api_key (str): API key for the Gemini API.

    Returns:
        List[str]: List of text chunks with context.
    """
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

        # Define the prompt for each chunk
        prompt = f"""
        Extract essential diagnostic information from this medical wiki page to assist in diagnosing based on visual or text prompts:
        {chunk}
        """

        # Prepare the request data
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json"
        }

        # Make the API call for each chunk
        response = requests.post(f"{url}?key={api_key}", headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            response_data = response.json()

            # Extract and process the context from the response
            summary = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
            return summary
        else:
            # If there is an error, return the original chunk
            print(f"API call failed with status code {response.status_code}")
            print((f"Reponse: {response.text}"))
            return chunk

    except Exception as e:
        raise Exception(f"Contextualize chunks failed: {e}")
