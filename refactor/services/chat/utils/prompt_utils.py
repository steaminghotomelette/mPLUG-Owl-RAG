from typing import Generator

# --------------------------------------------
# Utility Functions
# --------------------------------------------
def filter_stream(original_generator: Generator) -> Generator:
    for value in original_generator:
        if "<|im_start|>assistant" in value:
            continue
        else:
            yield value