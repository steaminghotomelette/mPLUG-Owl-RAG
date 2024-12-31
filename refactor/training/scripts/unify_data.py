import json

def read_json_file(file_path):
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: Parsed JSON content, expected to be a list.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(data, output_path):
    """
    Write JSON data to a file with consistent formatting and Unix-style line endings.
    
    Args:
        data (list): Data to write to the file.
        output_path (str): Path to the output JSON file.
    """
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Initialize empty list for all data
all_data = []

# List of input JSON file paths
file_paths = [
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/BioASQ/bio_asq_train_data.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/iu_xray/iu_xray_train_data.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/MMLU-Med/mmlu_med_train_data.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/PMC-VQA/pmc_vqa_1.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/PMC-VQA/pmc_vqa_2.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/PMC-VQA/pmc_vqa_3.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/PMC-VQA/pmc_vqa_4.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/brazil/output.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/israel/output.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/japan/output.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/V/spain/output.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/vqa-rad/vqa_rad_1/vqa_rad_1.json',
    'C:/Users/ILLEGEAR/personal-projects/mPLUG-Owl-RAG/refactor/training/scripts/vqa-rad/vqa_rad_2/vqa_rad_2.json',
]

# Read and combine data from all files
for file_path in file_paths:
    try:
        all_data.extend(read_json_file(file_path))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Save merged data
output_path = 'output.json'
try:
    write_json_file(all_data, output_path)
    print(f"Total entries in merged file: {len(all_data)}")
except Exception as e:
    print(f"Error writing merged file: {e}")
