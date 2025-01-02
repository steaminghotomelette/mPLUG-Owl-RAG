import json
from typing import Dict

def process_row(row: Dict[str, str]) -> Dict[str, str]:

    history = []
    for snippet in row["snippets"]:
        history.append([snippet["text"], "Understood as context."])
    
    # Return the formatted dictionary
    return {
        "query": row["body"],
        "response": row["ideal_answer"][0],
        "history": history,
        "images": [],
    }

with open("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/BioASQ/training11b.json", "r") as f:
    data = json.load(f)
    data = data["questions"]
    data = [process_row(row) for row in data]

    with open("bio_asq_train_data.json", "w") as f:
        json.dump(data, f, indent=4)
