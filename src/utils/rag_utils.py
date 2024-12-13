import lancedb
from db.connection import DBConnection
from db.utils import DOC_COLLECTION_NAME, MM_COLLECTION_NAME, USER_COLLECTION_NAME
from utils.embed_utils import EmbeddingModelManager, EmbeddingModel


class RAGManager():
    def __init__(self):
        self.embedding_model_manager = EmbeddingModelManager()
        self.db = DBConnection()
        self.image_text_table = None
        self.doc_table = None
        self.user_table = None
        self.load_collections()

    def load_collections(self):
        """
        Create or load required multimodal, document and user collections
        """
        self.image_text_table = self.db.create_or_load_collection(
            MM_COLLECTION_NAME, self.embedding_model_manager
        )
        self.doc_table = self.db.create_or_load_collection(
            DOC_COLLECTION_NAME, self.embedding_model_manager
        )
        self.user_table = self.db.create_or_load_collection(
            USER_COLLECTION_NAME, self.embedding_model_manager
        )

    def is_model_changed(self, embedding_model: str):
        """
        Returns true when embedding_model input differ from 
        what is being used in embedding model manager.
        """
        return embedding_model != self.embedding_model_manager.model_type.value

    def update_model(self, embedding_model: str):
        """
        Update embedding model and RAG target collection upon embedding model change
        """
        if self.is_model_changed(embedding_model):
            self.embedding_model_manager.set_model(
                EmbeddingModel(embedding_model))
            self.load_collections()

    def reset_user_table(self):
        """
        Reset RAG by dropping all collections of user table for all embedding models.
        """
        collections = self.db.list_collections()
        dropped = 0
        for model in EmbeddingModel:
            name = f"{USER_COLLECTION_NAME}_{model.value}"
            if name in collections:
                self.db.drop_collection(name)
                dropped += 1

        self.load_collections()
        return {"success": dropped > 0, "collections_dropped": dropped}
