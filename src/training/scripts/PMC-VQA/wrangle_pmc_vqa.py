import pandas as pd
from typing import Dict, List
import json
import re

def clean_mcq_option(text: str) -> str:
    # Remove any leading/trailing whitespace first
    text = text.strip()
    # Remove patterns like "A:", "B:", " A:", " B:" etc.
    cleaned = re.sub(r'^\s*[A-D]\s*:', '', text).strip()
    return cleaned

def process_row1(row: Dict[str, str]) -> Dict[str, str]:
    # Get the correct answer text based on the Answer_label    
    return {
        "query": f'<|image|><|image|>{row["Question"].strip()}',
        "response": str(row["Answer"]),
        "images": [f"images/{row['Figure_path']}", f"images/{row['Figure_path']}"]
    }

def process_row2(row: Dict[str, str]) -> Dict[str, str]:
    # Get the correct answer text based on the Answer_label
    answer_label = row["Answer"].strip()
    answer_map = {
        'A': row['Choice A'].strip(),
        'B': row['Choice B'].strip(),
        'C': row['Choice C'].strip(),
        'D': row['Choice D'].strip()
    }
       
    return {
        "query": f'<|image|><|image|>{row["Question"].strip()}',
        "response": clean_mcq_option(answer_map[answer_label]),
        "images": [f"images/{row['Figure_path']}", f"images/{row['Figure_path']}"]
    }

def process_db(row: Dict[str, str]) -> Dict[str, str]:
    # Get the correct answer text based on the Answer_label
    answer_label = row["Answer"].strip()
    answer_map = {
        'A': row['Choice A'].strip(),
        'B': row['Choice B'].strip(),
        'C': row['Choice C'].strip(),
        'D': row['Choice D'].strip()
    }
       
    return {
        "context": f'Caption:{row['Caption'].strip()}\nQuestion:{row["Question"].strip()}\nAnswer:{clean_mcq_option(answer_map[answer_label])}\n',
        "image": f"{row['Figure_path']}"
    }

def db_convert_csv_to_json(csv_path: str, output_prefix: str, row_proccessor_fn, chunk_size = 10000) -> None:
    # Read and process the CSV file in chunks
    for chunk_index, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        # Process each row in the chunk
        processed_data = [row_proccessor_fn(row) for _, row in chunk.iterrows()]
        
        # Write the chunk to a JSON file
        output_path = f"{output_prefix}_{chunk_index + 1}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        

def convert_csv_to_json(csv_path: str, output_path: str, row_proccessor_fn) -> None:
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert each row to the desired format
    processed_data = [row_proccessor_fn(row) for _, row in df.iterrows()]
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    # convert_csv_to_json("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/PMC-VQA/train.csv", "pmc_vqa_1.json", process_row1)
    # convert_csv_to_json("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/PMC-VQA/test.csv", "pmc_vqa_2.json", process_row1)
    db_convert_csv_to_json("C:/Users/user/Desktop/Y3/FYP/mPLUG-Owl-RAG/src/training/datasets/Medical/train_2.csv", "pmc_vqa_rag", process_db)
    # convert_csv_to_json("C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/datasets/Medical/PMC-VQA/test_2.csv", "pmc_vqa_4.json", process_row2)
