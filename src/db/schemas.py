import pyarrow as pa
import lancedb
import numpy as np
from typing import Dict, Any

def create_schema(db_path: str, dimension: int) -> pa.Schema:
    """ 
    Creates a schema for a collection of vectors embedded with CLIP using LanceDB.
    
    Args:
        db_path (str): The file path where the LanceDB database will be stored.
    
    Returns:
        lancedb.LanceTable: A configured LanceDB table for CLIP embeddings.
    """
    
    # --------------------------------------------
    # Define Schema using PyArrow
    # --------------------------------------------
    schema = pa.schema([
        pa.field('id', pa.int64()),  # Primary key
        pa.field('embedding', pa.list_(pa.float32(), dimension)),  # Fixed-size embedding vector
        pa.field('metadata', pa.string()),  # Metadata as string
        pa.field('storage_link', pa.string())  # Link to storage location
    ])
    
    return schema