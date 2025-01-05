import pyarrow as pa
from enum import Enum

_concat_tables_args = {"promote_options": "default"}

# --------------------------------------------
# Enum
# --------------------------------------------
class Domain(Enum):
    """
    Enum class for domains supported
    """
    MEDICAL     = "Medical"
    FORENSICS   = "Forensics"
    DEFAULT     =  MEDICAL

class EmbeddingModel(Enum):
    """
    Enum class for embedding models supported
    """
    BLIP    = "BLIP"
    DEFAULT =  BLIP

# --------------------------------------------
# Results processing
# --------------------------------------------
def combine_results(user: pa.Table, multimodal: pa.Table, docs: pa.Table) -> pa.Table:
    """
    Combine the results of searching from user table, document table and 
    multimodal table.

    Args:
        user (pa.Table): PyArrow table of user table search result
        multimodal (pa.Table): multimodal table search result
        docs (pa.Table): document table search result

    Returns:
        pa.Table: PyArrow table of sorted combined results.
    """
    try:
        res = pa.concat_tables([user, multimodal, docs], **_concat_tables_args)
        if res.num_rows > 0:
            res = res.sort_by([("_relevance_score", "descending")])
        return res
    except Exception as e:
        raise Exception(f"Failed combining results: {e}")
    
def deduplicate(table: pa.Table):
    """
    Deduplicate the table based on the `id` and `text` column.
    """
    try:
        if isinstance(table, list):
            table = pa.Table.from_pylist(table)
        
        if table.num_rows > 0:
            df = table.to_pandas()
            # Deduplicate based on 'title' and 'text' if 'title' exists
            if 'title' in df.columns and 'text' in df.columns:
                df = df.drop_duplicates(subset=['title', 'text'])

            # Deduplicate based on 'id' and 'text' if they exist
            elif 'id' in df.columns and 'text' in df.columns:
                df = df.drop_duplicates(subset=['id', 'text'])

            # Convert back to PyArrow Table
            deduplicated_table = pa.Table.from_pandas(df)
            return deduplicated_table
        return table
    except Exception as e:
        raise Exception(f"Failed deduplication: {table.schema} {e}")
    