from pymilvus import MilvusClient
import db.schemas

def create_CLIP_collection(client: MilvusClient, collection_name: str) -> None:
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
    clip_schema, clip_index = db.schemas.create_CLIP_schema(client)

    # --------------------------------------------
    # Create the Collection in Milvus
    # --------------------------------------------
    client.create_collection(
        collection_name=collection_name,
        schema=clip_schema,
        index_params=clip_index
    )
