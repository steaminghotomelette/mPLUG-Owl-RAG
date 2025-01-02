import numpy as np
import pyarrow as pa
from enum import Enum

WEIGHT = {
    "USER": 0.4,
    "MULTIMODAL": 0.4,
    "DOCUMENT": 0.2
}

THRESHOLD = {
    "USER": 0.7,
    "MULTIMODAL": 0.7,
    "DOCUMENT": 0.7
}

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
    CLIP    = "CLIP"
    DEFAULT =  BLIP


# --------------------------------------------
# Results processing
# --------------------------------------------
def _weight_table(table: pa.Table, weight: float) -> pa.Table:
    """
    Weighs relevance score in table according to weight given.
    Weight is expected to be in float format, i.e. 0.6 for 60%

    Args:
        table (pa.Table): PyArrow table to weight
        weight (float): Weight to apply for relevance score (i.e. 0.6 for 60%)

    Returns:
        pa.Table: PyArrow table with new '_weighted_relevance_score' column.
    """
    try:
        if not isinstance(table, pa.Table):
            table = table.to_arrow()

        if "_rowid" not in table.column_names:
            table = table.append_column("_rowid", pa.array(range(len(table)), type=pa.uint64()))

        if "_relevance_score" in table.column_names:
            weighted_scores = [weight * score for score in table["_relevance_score"].to_pylist()]
            table = table.append_column("_weighted_relevance_score", pa.array(weighted_scores))
        return table
    except Exception as e:
        raise Exception(f"Failed creating weighted table: {e}")

def combine_results(user: pa.Table,
                    multimodal: pa.Table,
                    docs: pa.Table,
                    w_user: float = WEIGHT['USER'],
                    w_mm: float = WEIGHT['MULTIMODAL'],
                    w_doc: float = WEIGHT['DOCUMENT']) -> pa.Table:
    """
    Combine the results of searching from user table, document table and 
    multimodal table.
    Apply weightage for each result, and reranks them based on the new
    weighted relevance score in descending order.

    Args:
        user (pa.Table): PyArrow table of user table search result
        multimodal (pa.Table): multimodal table search result
        docs (pa.Table): document table search result
        w_user (float): weight to apply for user search result
        w_mm (float): weight to apply for multimodal search result
        w_doc (float): weight to apply for document search result

    Returns:
        pa.Table: PyArrow table of sorted combined results.
    """
    try:
        u = _weight_table(user, w_user)
        m = _weight_table(multimodal, w_mm)
        d = _weight_table(docs, w_doc)
        res = pa.concat_tables([u, m, d], **_concat_tables_args)
        res = res.sort_by(
            [("_weighted_relevance_score", "descending")]
        )
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

        if 'id' in table.column_names and 'text' in table.column_names:
            row_id = np.array(table.column("id"))
            row_text = np.array(table.column("text"))

            # deduplicate
            mask = np.full((table.shape[0]), False)
            combined = np.array(list(zip(row_id.tolist(), row_text.tolist())))
            _, mask_indices = np.unique(combined, axis=0,return_index=True)
            return table.take(mask_indices)
        
        return table
    except Exception as e:
        raise Exception(f"Failed deduplication: {table.schema} {e}")
    
def format_query(query: str, tag: str) -> str:
    """Formats the query based on the type of search.

    Args:
        query (str): The query string.
        type (str): The type of search (e.g., "text", "image").

    Returns:
        str: The formatted query string.
    """
    if tag in ["image", "video"]:
        return f"<|{tag}|><|{tag}|>{query}"
    else:
        raise ValueError(f"Invalid search type: {type}")