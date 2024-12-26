from typing import Generator

# --------------------------------------------
# Utility Functions
# --------------------------------------------
def filter_stream(generator: Generator) -> Generator:
    buffer = ""
    latest_response = ""
    user_marker = "<|im_start|>user"
    assistant_marker = "<|im_start|>assistant"
    
    for chunk in generator:
        buffer += chunk
        
        # If we see a new user message, clear everything before it
        if user_marker in buffer:
            buffer = buffer.split(user_marker)[-1]
            latest_response = ""
            
        # If we see an assistant marker, get ready to yield new content
        if assistant_marker in buffer:
            # Get only the content after the most recent assistant marker
            parts = buffer.split(assistant_marker)
            latest_response = parts[-1]
            buffer = parts[-1]
            yield latest_response
            continue
            
        # If we're actively streaming a response, yield any new content
        if latest_response and chunk:
            yield chunk