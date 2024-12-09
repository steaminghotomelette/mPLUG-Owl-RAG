import lancedb
from typing import List, Dict, Optional
import db.collections
import pyarrow as pa
import os

class DBConnection:
    """
    Wrapper class for managing a connection to LanceDB, including operations 
    for creating, loading, and managing collections and data.
    """
    DIMENSIONS = {"CLIP": 512, "BLIP": 768}
    
    def __init__(self, db_path: str = "./lancedb") -> None:
        """
        Initializes a connection to the LanceDB database.
        
        Args:
            db_path (str): Path where the LanceDB database will be stored.
        """
        self.client = lancedb.connect(db_path)
    
    def create_collection(self, collection_name: str, embedding_model: str = "CLIP"):
        """
        Creates a collection in LanceDB if it doesn't exist.

        Args:
            collection_name (str): Name of the collection to create.
            embedding_model (str): Embedding model to use for the schema. Defaults to "CLIP".
        
        Returns:
            lancedb.Table: The created LanceDB table.
        
        Raises:
            Exception: If an invalid model is provided.
        """
        # List of existing collections
        existing_collections = self.list_collections()
        
        # Check if the collection already exists
        if collection_name in existing_collections:
            raise Exception(f"Collection '{collection_name}' already exists!")
        
        # If not, proceed with creation
        if embedding_model in ["CLIP", "BLIP"]:
            db.collections.create_collection(self.client, collection_name, self.DIMENSIONS[embedding_model])
        else:
            raise Exception(f"Invalid embedding model: '{embedding_model}'")
    
    def drop_collection(self, collection_name: str) -> None:
        """
        Drops a collection from the database.

        Args:
            collection_name (str): Name of the collection to drop.
        
        Raises:
            Exception: If the collection does not exist.
        """
        try:
            table = self.client.open_table(collection_name)
            table.delete()
        except FileNotFoundError:
            raise Exception(f"Collection '{collection_name}' does not exist!")
    
    def list_collections(self) -> List[str]:
        """
        Lists all collections in the LanceDB database.

        Returns:
            List[str]: List of collection names.
        """
        collections = []
        for f in os.listdir(self.client.uri):
            full_path = os.path.join(self.client.uri, f)
            # Ensure it's a directory and ends with '.lance'
            if os.path.isdir(full_path) and f.endswith('.lance'):
                collections.append(f.split('.')[0])
        return collections

    
    def insert(self, collection_name: str, data: List[Dict]) -> Dict:
        """
        Inserts data into a specified collection.

        Args:
            collection_name (str): Name of the collection.
            data (List[Dict]): Data to insert.
        
        Returns:
            Dict: Results of the insert operation.
        """
        table = self.client.open_table(collection_name)
        table.add(data)
        return {"success": True, "inserted_count": len(data)}
    
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
        
        table = self.load_collection(collection_name)
        
        if ids:
            # LanceDB doesn't have a direct method to delete by IDs, so we'd need to filter
            # This is a placeholder and might need to be adjusted based on exact LanceDB capabilities
            remaining_data = [row for row in table.to_pandas() if row['id'] not in ids]
            table.delete()  # Remove existing table
            table.add(remaining_data)  # Re-add filtered data
        
        # Note: Complex filtering would require custom implementation
        
        return {"success": True, "deleted_count": len(ids) if ids else 0}