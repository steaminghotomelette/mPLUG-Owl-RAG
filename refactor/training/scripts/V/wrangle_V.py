import pandas as pd
import json
import os
import base64
from typing import Dict, List, Union

def safe_strip(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def save_base64_image(base64_str: str, output_dir: str, index: int, offset: str) -> str:
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
   
    # Create filename
    image_filename = f"image_{index}_{offset}.png"
    image_path = os.path.join(output_dir, image_filename)
   
    # Decode and save image
    try:
        image_data = base64.b64decode(base64_str)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        return image_filename
    except Exception as e:
        print(f"Error saving image {index}: {e}")
        return None

def process_row(row: pd.Series, output_dir: str, index: int, offset: str) -> Dict[str, Union[str, List[str]]]:
    """Process a single row from the TSV file"""
    image_path = []
   
    # Save image if it exists
    if pd.notna(row['image']) and row['image'].strip():
        saved_path = save_base64_image(row['image'], output_dir, index, offset)
        if saved_path:
            # Use forward slashes and normalize the path
            normalized_path = os.path.normpath(f"{output_dir}/{saved_path}").replace('\\', '/')
            image_path = [normalized_path]

    # Get the correct answer text
    answer = ""
    if pd.notna(row['correct_option']):
        option_map = {
            'A': safe_strip(row['A']),
            'B': safe_strip(row['B']),
            'C': safe_strip(row['C']),
            'D': safe_strip(row['D'])
        }
        answer = option_map.get(safe_strip(row['correct_option']), "")

    query_string = f"<|image|>{safe_strip(row['question'])}"
    for key in option_map.keys():
        query_string += f'\n{key}: {safe_strip(row[key])}'
        
    return {
        "query": query_string,
        "response": f"{row['correct_option']}: {answer}",
        "images": [f"images/image_{index}_{offset}.png"]
    }

def convert_tsv_to_json(input_path: str, output_dir: str, offset: str) -> None:
    """
    Convert TSV file to JSON format and save images.
    """
    # Normalize paths with forward slashes
    output_dir = os.path.normpath(output_dir).replace('\\', '/')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
   
    # Create images subdirectory
    images_dir = f"{output_dir}/images"
    os.makedirs(images_dir, exist_ok=True)
   
    # Read the TSV file
    df = pd.read_csv(input_path, sep='\t')
   
    # Process each row
    processed_data = []
    for idx, row in df.iterrows():
        try:
            processed_row = process_row(row, images_dir, idx, offset)
            processed_data.append(processed_row)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
   
    # Write to JSON file
    output_json = f"{output_dir}/output.json"
    with open(output_json, 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    convert_tsv_to_json(
        input_path="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/V/brazil_english_processed.tsv",
        output_dir="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/brazil",
        offset="brazil"
    )
    convert_tsv_to_json(
        input_path="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/V/israel_english_processed.tsv",
        output_dir="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/israel",
        offset="israel"
    )
    convert_tsv_to_json(
        input_path="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/V/japan_english_processed.tsv",
        output_dir="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/japan",
        offset="japan"
    )
    convert_tsv_to_json(
        input_path="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/V/spain_english_processed.tsv",
        output_dir="C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/spain",
        offset="spain"
    )
    