import time
import requests
import json


def contextualize_chunks(chunks, api_key):
    """
    Uses the Gemini API to provide a context for each chunk of text.

    Args:
        chunks (List[str]): Text chunks (when combined, comprise the entire document).
        api_key (str): API key for the Gemini API.

    Returns:
        List[str]: List of text chunks with context.
    """
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        document = " ".join(chunks)
        new_chunks = []

        for chunk in chunks:
            prompt = f"""
            <document>
            {document}
            </document>
            Here is the chunk we want to situate within the whole document
            <chunk>
            {chunk}
            </chunk>
            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk in a medical VQA context. Answer only with the succinct context and nothing else.
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

            response = requests.post(
                f"{url}?key={api_key}", headers=headers, data=json.dumps(data))
            time.sleep(1)
            if response.status_code == 200:
                response_data = response.json()
                context = response_data['candidates'][0]['content']['parts'][0]['text']
                if context:
                    new_chunk = f"{context} {chunk}"
                    new_chunks.append(new_chunk)
                else:
                    new_chunks.append(chunk)
            else:
                print(f"Error: {response.status_code}, {response.text}")

        return new_chunks
    except Exception as e:
        raise Exception(f"Contextualize chunks failed: {e}")
