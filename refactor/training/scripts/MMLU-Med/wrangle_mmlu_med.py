import json
from typing import Dict

def process_row(row: Dict[str, str]) -> Dict[str, str]:
    query_string = row["question"]
    for key in row["options"].keys():
        query_string += f'\n{key}: {row["options"][key]}'
    return {
        "query": query_string,
        "response": f'{row["answer"]}: {row["options"][row["answer"]]}',
        "images": [],
    }

# Specify the correct encoding (e.g., utf-8) while opening the file
with open("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/MMLU-Med/US_qbank.jsonl", "r", encoding="utf-8") as f:
    # Read and parse the JSONL data
    data = [json.loads(line) for line in f]
    data = [process_row(row) for row in data]

    # Save the processed data
    with open("mmlu_med_train_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)