from io import BytesIO
import lancedb
from db.utils import MM_COLLECTION_NAME, USER_COLLECTION_NAME, DOC_COLLECTION_NAME
import db.schemas as schemas
from utils.embed_utils import EmbeddingModelManager
from datasets import load_dataset

# --------------------------------------------
# General Collection Creation
# --------------------------------------------


def create_image_text_collection(
    client: lancedb.db.DBConnection,
    collection_name: str,
    text_dimension: int,
    image_dimension: int
) -> lancedb.db.Table:
    """
    Creates a collection in the LanceDB database for storing text and image embeddings.

    This function initializes a collection with a schema that supports both text and image 
    embeddings. It ensures the collection is indexed for efficient searching and querying.

    Args:
        client (lancedb.db.DBConnection): The LanceDB client connection.
        collection_name (str): The name of the collection to be created.
        text_dimension (int): The dimension of the text embedding vectors.
        image_dimension (int): The dimension of the image embedding vectors.

    Returns:
        lancedb.db.Table: Table created
    """
    schema = schemas.create_image_text_schema(text_dimension=text_dimension,
                                              image_dimension=image_dimension)
    table = client.create_table(
        name=collection_name,
        schema=schema,
        mode="overwrite"
    )
    schemas.create_index(table)
    return table


def create_text_collection(
    client: lancedb.db.DBConnection,
    collection_name: str,
    text_dimension: int,
) -> lancedb.db.Table:
    """
    Creates a text-based collection in the LanceDB database with the specified parameters.

    This function initializes a collection for storing text embeddings. It uses a predefined 
    schema for text data and ensures an index is created for efficient querying.

    Args:
        client (lancedb.db.DBConnection): The LanceDB client connection.
        collection_name (str): The name of the collection to be created.
        text_dimension (int): The dimension of the text embedding vectors.

    Returns:
        lancedb.db.Table: Table created

    """
    schema = schemas.create_text_schema(text_dimension=text_dimension)
    table = client.create_table(
        name=collection_name,
        schema=schema,
        mode="overwrite"
    )
    schemas.create_index(table)
    return table

# --------------------------------------------
# Specific Collection Creation
# --------------------------------------------


def create_multimodal_table(
    client: lancedb.db.DBConnection,
    embedding_model: EmbeddingModelManager
):
    """
    Function to create and populate multimodal table that stores image and text pairs.
    """
    table = create_image_text_collection(
        client, f"{MM_COLLECTION_NAME}_{embedding_model.model_type.name}",
        embedding_model.text_dimension,
        embedding_model.image_dimension
    )
    # populate table with dataset
    insert_vqa(table, embedding_model)
    return table


def create_doc_table(
    client: lancedb.db.DBConnection,
    embedding_model: EmbeddingModelManager
):
    """
    Function to create and populate documents table that stores documents and texts.
    TODO: Populate table with dataset (Document chunking)
    """
    table = create_text_collection(
        client, f"{DOC_COLLECTION_NAME}_{embedding_model.model_type.name}",
        embedding_model.text_dimension
    )
    return table


def create_user_table(
    client: lancedb.db.DBConnection,
    embedding_model: EmbeddingModelManager
):
    """
    Function to create multimodal table for user to upload documents.
    """
    table = create_image_text_collection(
        client, f"{USER_COLLECTION_NAME}_{embedding_model.model_type.name}",
        embedding_model.text_dimension,
        embedding_model.image_dimension
    )
    return table

# --------------------------------------------
# Populate collection
# --------------------------------------------


# Batch data insertion size
BATCH_SIZE = 128


def insert_vqa(table: lancedb.db.Table,
               embedding_model: EmbeddingModelManager):
    """
    Inserts VQA (Visual Question Answering) data (image + text) into the specified table.
    Data source: flaviagiammarino/vqa-rad

    Args:
        table (lancedb.db.Table): The table where the VQA data will be inserted. 
                                  The table must have 'image_embedding' and 'text_embedding' fields.
        embedding_model (EmbeddingModelManager): An instance of the embedding model manager 
                                                 responsible for embedding images and text.

    Raises:
        Exception: If the table schema does not contain the required 'image_embedding' or 
                   'text_embedding' fields, or if there is a failure during data insertion.

    Returns:
        None: This function does not return any value. It directly inserts data into the table.
    """
    cols = table.schema.names
    if "image_embedding" not in cols:
        raise Exception(f"Table must have 'image_embedding' field")
    if "text_embedding" not in cols:
        raise Exception(f"Table must have 'text_embedding' field")

    data = load_dataset("flaviagiammarino/vqa-rad")
    data = data['train'][:5]
    batch_data = []

    for image, q, a in zip(data['image'], data['question'], data['answer']):
        text = f"question: {q}, answer: {a}"
        metadata = None
        bytes_io = BytesIO()
        image.save(bytes_io, format="JPEG")
        img_bytes = bytes_io.getvalue()
        img_embbeding, raw_image = embedding_model.embed_image(img_bytes)
        text_embedding = embedding_model.embed_text(text)

        batch_data.append(
            {
                "text": text,
                "text_embedding": text_embedding,
                "image_embedding": img_embbeding,
                "image_data": raw_image,
                "metadata": metadata
            }
        )
        if len(batch_data) % BATCH_SIZE == 0:
            try:
                table.add(batch_data)
                batch_data = []
            except Exception as e:
                raise Exception(f"Failed inserting vqa dataset: {e}")

    if len(batch_data) != 0:
        try:
            table.add(batch_data)
        except Exception as e:
            raise Exception(f"Failed inserting vqa dataset: {e}")
