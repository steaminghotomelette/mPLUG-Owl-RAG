from typing import Generator, Dict, List, Tuple, Any
from langchain.prompts import PromptTemplate
from PIL import Image
from io import BytesIO
import base64

def filter_stream(original_generator: Generator) -> Generator:
    """Filters out specific strings from a generator stream."""
    for value in original_generator:
        if "<|im_start|>assistant" in value:
            continue
        else:
            yield value

def create_system_prompt(domain: str) -> str:
    """Constructs a system prompt based on the domain."""
    rag_prompt = PromptTemplate(
        input_variables=["domain"],
        template=(
            "You are a knowledgeable {domain} expert. Your task is to answer a "
            "question using the provided external documents. Note that some of these documents "
            "may be noisy or irrelevant to the question.\n\n"
            "Instructions:\n"
            "1. Carefully analyze all provided documents.\n"
            "2. Identify which documents are relevant and which are irrelevant to the question.\n"
            "3. Think through the problem step-by-step, using only the relevant documents to determine the correct answer.\n\n"
            "4. Always explain your answers in great detail, never provide one word or short answers.\n\n"
            "Your responses will be used for research purposes only, so please provide a definite answer. "
            "If all of the retrieved documents are irrelevant, you have two options:\n"
            "1. Answer based on your own knowledge if you are absolutely certain.\n"
            "2. Refuse to answer by replying that you have insufficient information."
        )
    )
    return rag_prompt.format(domain=domain)

def create_image_based_prompt(
    user_query: str, retrieved_context: Dict[str, Any]
) -> Tuple[List[Image.Image], str]:
    """Creates a prompt for image-based queries with associated context."""
    formatted_context = "" if len(retrieved_context.get("content", [])) > 0 else "\nNo context retrieved."
    images = []

    for context_chunk in retrieved_context.get("content", []):
        if context_chunk.get("image_data", None) is not None:
            formatted_context += f"\n- <|image|>{context_chunk.get('text', '')}"
            try:
                image_data = base64.b64decode(context_chunk["image_data"])
                images.append(Image.open(BytesIO(image_data)).convert("RGB"))
            except Exception as e:
                raise ValueError(f"Error decoding image data: {e}")
        else:
            formatted_context += f"\n- {context_chunk.get('text', '')}"

    rag_prompt = PromptTemplate(
        input_variables=["user_query", "retrieved_context"],
        template=(
            "Here are the relevant documents:"
            "{retrieved_context}\n\n"
            "Here is the question:\n"
            "{user_query}\n\n"
            "Please think step-by-step and provide a clear and descriptive answer to the question, explain your answer in detail and do not repeat yourself."
        )
    )
    return (images, rag_prompt.format(user_query=user_query, retrieved_context=formatted_context))

def create_video_based_prompt(user_query: str, retrieved_context: Dict[str, Any]) -> str:
    """Creates a prompt for video-based queries using relevant context."""
    formatted_context = "" if len(retrieved_context.get("content", [])) > 0 else "\nNo context retrieved."
    for context_chunk in retrieved_context.get("content", []):
        formatted_context += f"\n- {context_chunk.get('text', '')}"

    rag_prompt = PromptTemplate(
        input_variables=["user_query", "retrieved_context"],
        template=(
            "Here are the relevant documents:"
            "{retrieved_context}\n\n"
            "Here is the question:\n"
            "{user_query}\n\n"
            "Please think step-by-step and provide a clear and descriptive answer to the question, explain your answer in detail and do not repeat yourself."
        )
    )
    return rag_prompt.format(user_query=user_query, retrieved_context=formatted_context)