import lancedb
from tqdm import tqdm
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
    
    match domain:
        case Domain.MEDICAL:
            insert_med_mm(table, embedding_model)
        case Domain.FORENSICS:
            insert_for_mm(table, embedding_model)
    
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

    match domain:
        case Domain.MEDICAL:
            insert_med_docs(table, embedding_model)
        case Domain.FORENSICS:
            insert_for_docs(table, embedding_model)

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
# Populate Medical Collection
# --------------------------------------------
def insert_med_mm(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:
    pass


def insert_med_docs(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:
   pass



# --------------------------------------------
# Populate Forensics Collection
# --------------------------------------------
def insert_for_mm(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:
   pass
        

def insert_for_docs(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:
    pass