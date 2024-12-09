import lancedb
from typing import Union
import db.schemas as schemas

def create_collection(
    client: lancedb.db.DBConnection, 
    collection_name: str,
    dimension: int
) -> None:
    """ 
    Creates a standard collection for vectors embedded with CLIP.
    
    Args:
        client (MilvusClient): The Milvus client to use for collection creation.
        collection_name (str): The name of the collection to be created.
    
    Returns:
        None
    """

    # --------------------------------------------
    # Create Collection Schema and Index
    # --------------------------------------------
    schema = schemas.create_schema(client, dimension)

    # --------------------------------------------
    # Create the Collection in Milvus
    # --------------------------------------------
    client.create_table(
            name=collection_name,
            schema=schema,
            mode="overwrite"  # Replace existing table if it exists
    )
