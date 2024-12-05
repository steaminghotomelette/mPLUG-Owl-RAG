from pymilvus import MilvusClient
import db.collections
from typing import List, Dict, Optional

class DBConnection:
    """
    Wrapper class for managing a connection to Milvus, including operations 
    for creating, loading, releasing, and managing collections and data.
    """

    def __init__(self, host: str = "localhost", port: str = "19530") -> None:
        """
        Initializes a connection to the Milvus database.
        
        Args:
            host (str): Hostname or IP of the Milvus server.
            port (str): Port of the Milvus server.
        """
        self.client = MilvusClient(
            uri=f"http://{host}:{port}",
            token="root:Milvus"
        )
    
    def create_collection(self, collection_name: str, embedding_model: str = "CLIP") -> None:
        """
        Creates a collection in Milvus if it doesn't exist.

        Args:
            collection_name (str): Name of the collection to create.
            embedding_model (str): Embedding model to use for the schema. Defaults to "CLIP".
        
        Raises:
            Exception: If the collection already exists or an invalid model is provided.
        """
        if self.client.has_collection(collection_name):
            raise Exception(f"Collection '{collection_name}' already exists!")

        # Handle embedding model
        if embedding_model == "CLIP":
            db.collections.create_CLIP_collection(self.client, collection_name)
        else:
            raise Exception(f"Invalid embedding model: '{embedding_model}'")
    
    def load_collection(self, collection_name: str) -> None:
        """
        Loads a collection into memory if it exists.

        Args:
            collection_name (str): Name of the collection to load.
        
        Raises:
            Exception: If the collection does not exist.
        """
        if not self.client.has_collection(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist!")
        self.client.load_collection(collection_name=collection_name)
    
    def release_collection(self, collection_name: str) -> None:
        """
        Releases a collection from memory if it exists.

        Args:
            collection_name (str): Name of the collection to release.
        
        Raises:
            Exception: If the collection does not exist.
        """
        if not self.client.has_collection(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist!")
        self.client.release_collection(collection_name=collection_name)
    
    def drop_collection(self, collection_name: str) -> None:
        """
        Drops a collection from the database if it exists.

        Args:
            collection_name (str): Name of the collection to drop.
        
        Raises:
            Exception: If the collection does not exist.
        """
        if not self.client.has_collection(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist!")
        self.client.drop_collection(collection_name=collection_name)
    
    def list_collections(self) -> List[str]:
        """
        Lists all collections in the Milvus database.

        Returns:
            List[str]: List of collection names.
        """
        return self.client.list_collections()
    
    def insert(self, collection_name: str, data: List[Dict]) -> Dict:
        """
        Inserts data into a specified collection.

        Args:
            collection_name (str): Name of the collection.
            data (List[Dict]): Data to insert.
        
        Returns:
            Dict: Results of the insert operation.
        """
        return self.client.insert(collection_name=collection_name, data=data)
    
    def upsert(self, collection_name: str, data: List[Dict]) -> Dict:
        """
        Upserts data into a specified collection.
        (NOTE UPSERT ONLY WORKS IN MILVUS 2.5 ONWARDS)

        Args:
            collection_name (str): Name of the collection.
            data (List[Dict]): Data to upsert.
        
        Returns:
            Dict: Results of the upsert operation.
        """
        return self.client.upsert(collection_name=collection_name, data=data)
    
    def delete(self, collection_name: str, ids: Optional[List[int]] = None, filter: Optional[str] = None) -> Dict:
        """
        Deletes data from a specified collection.

        Args:
            collection_name (str): Name of the collection.
            ids (Optional[List[int]]): List of IDs to delete.
            filter (Optional[str]): Filter condition for deletion.
        
        Raises:
            Exception: If neither IDs nor a filter is specified.
        
        Returns:
            Dict: Results of the delete operation.
        """
        if ids is None and filter is None:
            raise Exception("Please specify either 'ids' or a 'filter' for deletion!")
        return self.client.delete(collection_name=collection_name, ids=ids, filter=filter)
    
    def flush(self, collection_name: str) -> None:
        """
        Flushes a specified collection to persist data.

        Args:
            collection_name (str): Name of the collection to flush.

        Raises:
            Exception: If the collection does not exist.
        """
        if not self.client.has_collection(collection_name):
            raise Exception(f"Collection '{collection_name}' does not exist!")
        self.client.flush(collection_name=collection_name)
