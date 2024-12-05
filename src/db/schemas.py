from pymilvus import MilvusClient, DataType, CollectionSchema
from pymilvus.milvus_client import IndexParams
from typing import Tuple

def create_CLIP_schema(client: MilvusClient) -> Tuple[CollectionSchema, IndexParams]:
    """ 
    Creates a standard schema and index for a collection of vectors embedded with CLIP.
    
    Args:
        client (MilvusClient): The Milvus client to use for schema and index creation.
    
    Returns:
        Tuple: A tuple containing the collection schema and index parameters.
    """
    
    # --------------------------------------------
    # Create Collection Schema
    # --------------------------------------------
    schema = client.create_schema(
        auto_id=True, 
        enable_dynamic_field=True
    )

    # Add fields to the schema
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="storage_link", datatype=DataType.VARCHAR, max_length=512)
    
    # --------------------------------------------
    # Prepare Index Parameters
    # --------------------------------------------
    index_params = client.prepare_index_params()

    # Define index for the "id" field
    index_params.add_index(
        field_name="id",
        index_type="STL_SORT"
    )

    # Define index for the "embedding" field with additional parameters
    index_params.add_index(
        field_name="embedding", 
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )
    
    # Return schema and index parameters
    return schema, index_params
