import lancedb
import db.schemas as schemas
from io import BytesIO
from db.constants import MM_COLLECTION_NAME, USER_COLLECTION_NAME, DOC_COLLECTION_NAME, BATCH_SIZE
from datasets import load_dataset
from utils.rag_utils import Domain
from utils.embed_utils import EmbeddingModelManager

# --------------------------------------------
# General Collection Creation
# --------------------------------------------
def create_image_text_collection(
        client: lancedb.db.DBConnection,
        collection_name: str,
        text_dimension: int,
        image_dimension: int
) -> lancedb.db.Table:
    schema = schemas.create_image_text_schema(
        text_dimension=text_dimension,
        image_dimension=image_dimension
    )
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
        text_dimension: int
) -> lancedb.db.Table:
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
        embedding_model: EmbeddingModelManager,
        domain: Domain
) -> lancedb.db.Table:
    table = create_image_text_collection(
        client,
        f"{MM_COLLECTION_NAME}_{embedding_model.model_type.name}_{domain.value}",
        embedding_model.text_dimension,
        embedding_model.image_dimension
    )
    return table

def create_doc_table(
        client: lancedb.db.DBConnection,
        embedding_model: EmbeddingModelManager,
        domain: Domain
) -> lancedb.db.Table:
    table = create_text_collection(
        client,
        f"{DOC_COLLECTION_NAME}_{embedding_model.model_type.name}_{domain.value}",
        embedding_model.text_dimension
    )
    return table

def create_user_table(
        client: lancedb.db.DBConnection,
        embedding_model: EmbeddingModelManager,
        domain: Domain
) -> lancedb.db.Table:
    table = create_image_text_collection(
        client,
        f"{USER_COLLECTION_NAME}_{embedding_model.model_type.name}_{domain.value}",
        embedding_model.text_dimension,
        embedding_model.image_dimension
    )
    return table


# --------------------------------------------
# Populate Collection
# --------------------------------------------
def insert_vqa_rad(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:

    # Validate data
    if "image_embedding" not in table.schema.names:
        raise Exception("Table must have 'image_embedding' field")
    if "text_embedding" not in table.schema.names:
        raise Exception("Table must have 'text_embedding' field")
    
    # Load data
    data = load_dataset("flaviagiammarino/vqa-rad")
    data = data["train"][:5]
    batch_data = []

    # Prepare data batch
    for i, (image, question, answer) in enumerate(zip(data["image"], data["question"], data["answer"])):
        text = f"question: {question}, answer: {answer}"
        bytes_io = BytesIO()
        image.save(bytes_io, format="JPEG")
        image_bytes = bytes_io.getvalue()
        image_embedding, raw_image = embedding_model.embed_image(image_bytes)
        text_embedding = embedding_model.embed_text(text)

        batch_data.append(
            {
                "id": i + 1,
                "text": text,
                "text_embedding": text_embedding,
                "image_embedding": image_embedding,
                "image_data": raw_image,
                "metadata": None
            }
        )

        if len(batch_data) % BATCH_SIZE == 0:
            try:
                table.add(batch_data)
                batch_data = []
            except Exception as e:
                raise Exception(f"Failed inserting vqa dataset: {e}")
    
    # Handle leftovers
    if len(batch_data) != 0:
        try:
            table.add(batch_data)
        except Exception as e:
            raise Exception(f"Failed inserting vqa dataset: {e}")