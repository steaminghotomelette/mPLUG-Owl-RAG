import json
import os
import pandas as pd
from typing import Dict, Union, List

def inspect_image_data(row):
    image_data = row['image']
    print(f"Type of image data: {type(image_data)}")
    if isinstance(image_data, dict):
        print("Dictionary keys:", image_data.keys())
        print("Sample of values:", {k: type(v) for k, v in image_data.items()})
    return image_data

def save_image(image_data: dict, output_dir: str, index: int, offset: str) -> str:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename using the index
    image_filename = f"image_{index}_{offset}.png"
    image_path = os.path.join(output_dir, image_filename)
    
    # Convert the image data to numpy array
    # We'll need to modify this part based on the actual data structure
    if isinstance(image_data, dict) and 'bytes' in image_data:
        # If the image is stored as bytes
        with open(image_path, 'wb') as f:
            f.write(image_data['bytes'])
    else:
        raise ValueError(f"Unexpected image data format: {type(image_data)}")
    
    return image_filename

def process_row(row: pd.Series, output_dir: str, index: int, offset: str) -> Dict[str, Union[str, List[str]]]:
    image_path = []
    if pd.notna(row['image']):
        try:
            image_path = [save_image(row['image'], output_dir, index, offset)]
        except Exception as e:
            print(f"Error processing image at index {index}: {e}")
            # Optionally inspect the problematic data
            inspect_image_data(row)
    
    return {
        "query": f"<|image|>{str(row['question']).strip()}",
        "response": str(row['answer']).strip(),
        "images": [f"images/image_{index}_{offset}.png"]
    }

def convert_parquet_to_json(input_path: str, output_dir: str, output_file_name: str, offset: str) -> None:
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create images subdirectory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Read the parquet file
    df = pd.read_parquet(input_path)
    
    # Inspect first row's image data
    print("\nInspecting first row's image data:")
    if len(df) > 0:
        inspect_image_data(df.iloc[0])
    
    # Convert each row and save images
    processed_data = []
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            processed_row = process_row(row, images_dir, idx, offset)
            processed_data.append(processed_row)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Write to JSON file
    output_json = os.path.join(output_dir, output_file_name)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    convert_parquet_to_json(
        input_path="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/vqa-rad/data/train-00000-of-00001-eb8844602202be60.parquet",
        output_dir="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/vqa-rad/vqa_rad_1",
        output_file_name="vqa_rad_1.json",
        offset="a"
    )
    convert_parquet_to_json(
        input_path="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet",
        output_dir="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/vqa-rad/vqa_rad_2",
        output_file_name="vqa_rad_2.json",
        offset="b"
    )