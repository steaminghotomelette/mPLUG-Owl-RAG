import pyarrow as pa
import lancedb


def create_image_text_schema(text_dimension: int, image_dimension: int) -> pa.Schema:
    """ 
    Creates a schema for a collection of text and image vectors.

    Args:
        text_dimension (int): Dimension of text embedding
        image_dimension (int): Dimension of image embedding

    Returns:
        pa.Schema: Arrow Schema object for collection creation.
    """
    schema = pa.schema([
        pa.field('id', pa.int64()),
        # Fixed-size embedding vector
        pa.field('text', pa.string()),
        pa.field('text_embedding', pa.list_(pa.float32(), text_dimension)),
        pa.field('image_embedding', pa.list_(pa.float32(), image_dimension)),
        pa.field('image_data', pa.binary()),        # Raw binary data of image
        pa.field('metadata', pa.string()),
    ])
    return schema


def create_text_schema(text_dimension: int) -> pa.Schema:
    """ 
    Creates a schema for a collection of text vector.

    Args:
        text_dimension (int): Dimension of text embedding

    Returns:
        pa.Schema: Arrow Schema object for collection creation.
    """
    schema = pa.schema([
        pa.field('id', pa.int64()),
        pa.field('text', pa.string()),
        pa.field('text_embedding', pa.list_(pa.float32(), text_dimension)),
        pa.field('metadata', pa.string()),
    ])
    return schema


def create_index(table: lancedb.db.Table):
    """ 
    Creates index for LanceDB table columns 
    (text, text_embedding, image_embedding) for efficient searching.

    Args:
        table (lancedb.db.Table): The table for indexing.

    Returns:
        None
    """
    try:
        cols = table.schema.names
        # if "text_embedding" in cols:
        #   table.create_index("L2", "text_embedding")
        # if "image_embedding" in cols:
        #   table.create_index("L2", "image_embedding")
        if "text" in cols:
            table.create_fts_index("text",
                                   use_tantivy=False,
                                   with_position=True,
                                   replace=True)
    except Exception as e:
        print(f"Failed to create indexes: {e}")
        raise
