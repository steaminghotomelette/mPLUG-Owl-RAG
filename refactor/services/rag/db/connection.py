import os
import lancedb
import db.collections as collections
from db.constants import MM_COLLECTION_NAME, USER_COLLECTION_NAME, DOC_COLLECTION_NAME
from utils.rag_utils import Domain
from utils.embed_utils import EmbeddingModel, EmbeddingModelManager
from typing import List, Dict

class DBConnection:

    # --------------------------------------------
    # Initialization
    # --------------------------------------------
    def __init__(self, db_path: str = "./lancedb") -> None:
        self.client = lancedb.connect(db_path)
    
    # --------------------------------------------
    # Instance Methods
    # --------------------------------------------
    def list_collections(self) -> List[str]:
        collections = []
        for dir in os.listdir(self.client.uri):
            full_path = os.path.join(self.client.uri, dir)
            if os.path.isdir(full_path) and dir.endswith(".lance"):
                collections.append(dir.split(".")[0])
        return collections
    
    def create_or_load_collection(self, collection_name: str, embedding_model: EmbeddingModelManager, domain: Domain) -> None:

        # Get existing collections
        existing_collections = self.list_collections()

        # Check if collection already exists
        suffixed_collection_name = f"{collection_name}_{embedding_model.model_type.name}_{domain.value}"
        if suffixed_collection_name in existing_collections:
            return self.client.open_table(suffixed_collection_name)
        
        # If not, create the collection
        if embedding_model.model_type in EmbeddingModel:
            if collection_name == MM_COLLECTION_NAME:
                table = collections.create_multimodal_table(self.client, embedding_model, domain)
            elif collection_name == DOC_COLLECTION_NAME:
                table = collections.create_doc_table(self.client, embedding_model, domain)
            elif collection_name == USER_COLLECTION_NAME:
                table = collections.create_user_table(self.client, embedding_model, domain)
            else:
                raise Exception(f"Invalid collection name:'{collection_name}'")
            return table
        else:
            raise Exception(f"Invalid embedding model:'{embedding_model}'")
    
    def drop_collection(self, collection_name: str) -> None:
        self.client.drop_table(collection_name, ignore_missing=True)
    
    def count(self, collection_name: str) -> int:
        table = self.client.open_table(collection_name)
        return table.count_rows()
    
    def insert(self, collection_name: str, data: List[Dict]) -> Dict:
        # Open table and keep track of rows
        table = self.client.open_table(collection_name)
        before = table.count_rows()

        # Assign an ID to each entry
        for i in range(len(data)):
            data[i]["id"] = before + i + 1
        
        # Insert
        table.add(data)
        after = table.count_rows()

        if after > before:
            message = "File uploaded sucessfully!"
        else:
            message = "File upload failed!"
        
        return {
            "success": (after > before),
            "inserted_count": (after - before),
            "message": message
        }