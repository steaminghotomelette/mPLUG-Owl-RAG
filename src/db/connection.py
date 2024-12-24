import lancedb
from typing import List, Dict, Optional
import db.collections
import os
from db.utils import MM_COLLECTION_NAME, USER_COLLECTION_NAME, DOC_COLLECTION_NAME
from utils.embed_utils import EmbeddingModel, EmbeddingModelManager


class DBConnection:
    """
    Wrapper class for managing a connection to LanceDB, including operations
    for creating, loading, and managing collections and data.
    """

    def __init__(self, db_path: str = "./lancedb") -> None:
        """
        Initializes a connection to the LanceDB database.

        Args:
            db_path (str): Path where the LanceDB database will be stored.
        """
        self.client = lancedb.connect(db_path)

    def create_or_load_collection(self, collection_name: str,
                                  embedding_model: EmbeddingModelManager):
        """
        Load collection in LanceDB if it exist, else create collection.

        Args:
            collection_name (str): Name of the collection to create or load.
            embedding_model (str): Embedding model to use for the schema.

        Returns:
            lancedb.Table: LanceDB table created/loaded.

        Raises:
            Exception: If an invalid model is provided.
        """
        # List of existing collections
        existing_collections = self.list_collections()

        collection_name_model = f"{collection_name}_{embedding_model.model_type.name}"
        # Check if the collection already exists
        if collection_name_model in existing_collections:
            return self.client.open_table(collection_name_model)

        # If not, create collections
        if embedding_model.model_type in EmbeddingModel:
            if collection_name == MM_COLLECTION_NAME:
                table = db.collections.create_multimodal_table(
                    self.client, embedding_model)
            elif collection_name == USER_COLLECTION_NAME:
                table = db.collections.create_user_table(
                    self.client, embedding_model)
            elif collection_name == DOC_COLLECTION_NAME:
                table = db.collections.create_doc_table(
                    self.client, embedding_model)
            else:
                raise Exception(f"Invalid collection name:'{collection_name}'")
            return table
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
        self.client.drop_table(collection_name, ignore_missing=True)

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
    
    def count(self, collection_name: str) -> int:
        """
        Returns number of items in collection.

        Returns:
            int: count of items in collection.
        """
        table = self.client.open_table(collection_name)
        return table.count_rows()

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
        before = table.count_rows()
        # Assign id for each entry
        for i in range(len(data)):
            data[i]['id'] = before+i+1
        table.add(data)
        after = table.count_rows()
        if after > before:
            message = "File uploaded successfully!"
        else:
            message = "File upload failed..."
        return {"success": (after > before),
                "inserted_count": (after-before),
                "message": message}

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
            raise Exception(
                "Please specify either 'ids' or a 'filter' for deletion!")

        table = self.load_collection(collection_name)

        if ids:
            # LanceDB doesn't have a direct method to delete by IDs, so we'd need to filter
            # This is a placeholder and might need to be adjusted based on exact LanceDB capabilities
            remaining_data = [
                row for row in table.to_pandas() if row['id'] not in ids]
            table.delete()  # Remove existing table
            table.add(remaining_data)  # Re-add filtered data

        # Note: Complex filtering would require custom implementation

        return {"success": True, "deleted_count": len(ids) if ids else 0}
