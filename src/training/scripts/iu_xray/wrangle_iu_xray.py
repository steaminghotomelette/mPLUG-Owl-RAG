import json
from typing import Dict

def process_row(row: Dict[str, str]) -> Dict[str, str]:
    # Create image markers based on the number of images in the image_path
    image_markers = "".join([f"<|image|>" for _ in row["image_path"]])
    
    # Formulate the query
    query = f"{image_markers}Please describe the findings in the images."

    # Create image paths
    images = [("images/" + image_path) for image_path in row["image_path"]]
    
    # Return the formatted dictionary
    return {
        "query": query,
        "response": row["report"],
        "images": images,
    }

with open("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/iu_xray/annotation.json", "r") as f:
    data = json.load(f)
    flattened_data = data["train"] + data["val"] + data["test"]
    processed_data = [process_row(row) for row in flattened_data]

    with open("iu_xray_train_data.json", "w") as f:
        json.dump(processed_data, f, indent=4)
