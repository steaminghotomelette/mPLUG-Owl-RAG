from io import BytesIO
import json
import os
import lancedb
from tqdm import tqdm
import db.schemas as schemas
from db.constants import MM_COLLECTION_NAME, USER_COLLECTION_NAME, DOC_COLLECTION_NAME, BATCH_SIZE
from utils.rag_utils import Domain
from utils.embed_utils import EmbeddingModelManager
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.upload_utils import summarize_text, chunk_document
import io

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
    # Validate table
    if "image_embedding" not in table.schema.names:
        raise Exception("Table must have 'image_embedding' field")
    if "text_embedding" not in table.schema.names:
        raise Exception("Table must have 'text_embedding' field")
    
    try:
        files = sorted([f for f in os.listdir('db/seed/pmc_vqa') if f.startswith("pmc_vqa_rag_") and f.endswith(".json")])
        
        for filename in tqdm(files, total=len(files),
                            bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', 
                            desc=f'Inserting multimodal medical data:'):
            # Load data
            with open(f'db/seed/pmc_vqa/{filename}', 'r', encoding="utf-8", errors="ignore") as file:
                data = json.load(file)
            
            # Prepare path for images
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","..",".."))
            img_dir = os.path.join(base_dir, "images")

            # Prepare data batch
            batch_data = []
            before = table.count_rows()
            for i, record in tqdm(enumerate(data), total=len(data), 
                                  bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', 
                                  desc=f'Inserting file: {filename}'):
                # Obtain contexts
                text = f"{record['context']}"
                with open(os.path.join(img_dir, record['image']), "rb") as image_file:
                    image_data = BytesIO(image_file.read())

                # Generate embeddings
                image_embedding, raw_image = embedding_model.embed_image(image_data)
                text_embedding = embedding_model.embed_text(text)

                batch_data.append(
                    {
                        "id": before + i + 1,
                        "text": text,
                        "text_embedding": text_embedding,
                        "image_embedding": image_embedding,
                        "image_data": raw_image,
                        "metadata": None
                    }
                )

                if len(batch_data) % BATCH_SIZE == 0:
                    table.add(batch_data)
                    batch_data = []
            
            if len(batch_data) != 0:
                table.add(batch_data)

            # Indexing
            schemas.create_index(table)
            
    except Exception as e:
        raise Exception(f"Failed inserting medical multimodal dataset: {e}")

def insert_med_docs(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:
    try:
        df = pd.read_parquet('db/seed/MedWiki_HybridFilter.parquet')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

        # Process each document
        batch_data = []
        before = table.count_rows()
        for i, record in tqdm(df.iterrows(), total=len(df), desc='Processing medical wiki pages'):
            page_text = record['page_text']
            page_title = record['page_title']

            # Summarize the text to extract key medical details
            summarized_text = summarize_text(page_text, os.getenv("GEMINI_API_KEY"))
            chunks = text_splitter.split_text(summarized_text)
            
            # Process each chunk
            for chunk in chunks:
                text_embedding = embedding_model.embed_text(chunk)

                batch_data.append(
                    {
                        "id": before + i + 1,
                        "title": page_title,
                        "text": chunk,
                        "text_embedding": text_embedding,
                        "metadata": None
                    }
                )

                # Batch insert
                if len(batch_data) % 100 == 0:
                    table.add(batch_data)
                    batch_data = []

        # Insert remaining data
        if batch_data:
            table.add(batch_data)
        
        # Indexing
        schemas.create_index(table)
            
    except Exception as e:
        raise Exception(f"Failed inserting medical wiki pages: {e}")

# --------------------------------------------
# Populate Forensics Collection
# --------------------------------------------
def insert_for_mm(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:
    pass
        

def insert_for_docs(table: lancedb.db.Table, embedding_model: EmbeddingModelManager) -> None:
    try:
        with open("db/seed/Elements-of-Crimes.pdf", "rb") as document:
            document_stream = io.BytesIO(document.read())
            texts, metadata = chunk_document(document_stream, max_size=500, chunk_overlap=50)
        
        # Process each chunk
        batch_data = []
        before = table.count_rows()
        for i, text in tqdm(enumerate(texts), total=len(texts), desc='Processing forensics document'):
            text_embedding = embedding_model.embed_text(text)

            batch_data.append(
                {
                    "id": before + i + 1,
                    "text": text,
                    "title": None,
                    "text_embedding": text_embedding,
                    "metadata": str(metadata[i]["page"])
                }
            )
            
            # Batch insert
            if len(batch_data) % 20 == 0:
                table.add(batch_data)
                batch_data = []
                
        # Insert remaining data
        if batch_data:
            table.add(batch_data)
        
        # Indexing
        schemas.create_index(table)
        
    except Exception as e:
        raise Exception(f"Failed inserting forensics document: {e}")