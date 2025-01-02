from typing import Generator
from langchain.prompts import PromptTemplate

# --------------------------------------------
# Utility Functions
# --------------------------------------------
def filter_stream(original_generator: Generator) -> Generator:
    for value in original_generator:
        if "<|im_start|>assistant" in value:
            continue
        else:
            yield value

def create_system_prompt(domain: str) -> str:
    """
    Constructs a system prompt based on the domain.
    """
    rag_prompt = PromptTemplate(
        input_variables=["domain"],
        template=(
            """
            You are a knowledgeable {domain} expert. Your task is to answer a
            question using the provided external documents. Note that some of these documents
            may be noisy or irrelevant to the question.

            Instructions:
            1. Carefully analyze all provided documents.
            2. Identify which documents are relevant and which are irrelevant to the question.
            3. Think through the problem step-by-step, using only the relevant documents to determine the
            correct answer.
            
            Your responses will be used for research purposes only, so please provide a definite answer and format the output as a JSON object as instructed.
            If all of the retrieved documents are irrelevant, you have two options:
            1. Answer based on your own knowledge if you are absolutely certain.
            2. Refuse to answer by setting 'answer_choice' to 'insufficient information'.
            """
        )
    )
    # Fill in the template
    return rag_prompt.format(domain=domain)

def create_user_prompt(user_query: str, retrieved_chunks: str) -> str:
    """
    Constructs a system prompt using the query, and retrieved chunks.
    """
    formatted_context = format_context(retrieved_chunks)
    rag_prompt = PromptTemplate(
        input_variables=["user_query", "retrieved_context"],
        template=(
            """
            Here are the relevant documents:
            {retrieved_context}
            Here is the question:
            {user_query}
            Please think step-by-step and provide a clear and concise answer to the question.
            """
        )
    )
    # Fill in the template
    return rag_prompt.format(user_query=user_query, retrieved_context=formatted_context)


def format_context(chunks):
    """
    Formats a list of text chunks into a single string suitable for insertion into the prompt.
    """
    return "\n\n".join(f"- {chunk}" for chunk in chunks)