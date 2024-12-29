import json
from typing import Dict

# Load iu_xray dataset
with open("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/iu_xray/annotation.json", "r") as f:
    data = json.load(f)

# Split data into train and validation sets
train_data = data["train"]
valid_data = data["val"]
valid_data.extend(data["test"])

# Now process each entity in each dataset
def process_row(row: Dict[str, str]) -> Dict[str, str]:
    # Create image markers based on the number of images in the image_path
    image_markers = "".join([f"<|image|>" for _ in row["image_path"]])
    
    # Formulate the query
    query = f"{image_markers}Please describe the findings in the images."

    # Create image paths
    images = [("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/iu_xray/images/" + image_path) for image_path in row["image_path"]]
    
    # Return the formatted dictionary
    return {
        "query": query,
        "response": row["report"],
        "images": images,
    }

processed_train_data = [process_row(row) for row in train_data]
processed_val_data = [process_row(row) for row in valid_data]

# Save as wrangled datasets
with open("iu_xray_train_data.json", "w") as f:
    json.dump(processed_train_data, f, indent=4)

with open("iu_xray_valid_data.json", "w") as f:
    json.dump(processed_val_data, f, indent=4)