import base64
from datetime import datetime
from typing import List
import pyarrow as pa
from lancedb.rerankers import ColbertReranker, CrossEncoderReranker
from db.connection import DBConnection
from db.constants import DOC_COLLECTION_NAME, MM_COLLECTION_NAME, USER_COLLECTION_NAME
from utils.rag_utils import Domain, deduplicate, THRESHOLD, combine_results
from utils.embed_utils import EmbeddingModelManager, EmbeddingModel
from fastapi import UploadFile
from utils.upload_utils import chunk_document, contextualize_chunks
import os

class RAGManager():
    # --------------------------------------------
    # Initialisation
    # --------------------------------------------
    def __init__(self):
        self.embedding_model_manager = EmbeddingModelManager()
        self.db = DBConnection()
        self.reranker = ColbertReranker()
        self.image_text_table = None
        self.doc_table = None
        self.user_table = None
        self.domain = Domain.DEFAULT
        self.load_collections()

    # --------------------------------------------
    # Load collections
    # --------------------------------------------
    def load_collections(self):
        """
        Create or load required multimodal, document and user collections
        """
        self.image_text_table = self.db.create_or_load_collection(
            MM_COLLECTION_NAME, self.embedding_model_manager, self.domain
        )
        self.doc_table = self.db.create_or_load_collection(
            DOC_COLLECTION_NAME, self.embedding_model_manager, self.domain
        )
        self.user_table = self.db.create_or_load_collection(
            USER_COLLECTION_NAME, self.embedding_model_manager, self.domain
        )

    # --------------------------------------------
    # Embedding model and domain related functions
    # --------------------------------------------
    def is_model_changed(self, embedding_model: str, domain: str):
        """
        Returns true when embedding_model input differ from 
        what is being used in embedding model manager.
        """
        return (embedding_model != self.embedding_model_manager.model_type.value)\
                or (domain != self.domain.value)

    def switch_domain(self, embedding_model: str, domain: str):
        """
        Update embedding model and RAG target collection upon embedding model change
        """
        if self.is_model_changed(embedding_model, domain):
            self.domain = Domain(domain)
            self.load_collections()

    # --------------------------------------------
    # Reset user collection
    # --------------------------------------------
    def reset_user_table(self):
        """
        Reset RAG by dropping all collections of user table for all embedding models.
        """
        collections = self.db.list_collections()
        dropped = 0
        for model in EmbeddingModel:
            name = f"{USER_COLLECTION_NAME}_{model.value}_{self.domain.value}"
            if name in collections:
                self.db.drop_collection(name)
                dropped += 1

        self.load_collections()
        return {"success": dropped > 0, "collections_dropped": dropped}
    
    # --------------------------------------------
    # Upload to user collection
    # --------------------------------------------
    async def upload(self, files: List[UploadFile], metadata: List[str], embedding_model: str, domain: str):
        data = []
        try:
            self.switch_domain(embedding_model, domain)

            collection_name = f"{USER_COLLECTION_NAME}_{self.embedding_model_manager.model_type.value}_{self.domain.value}"

            for i, file in enumerate(files):
                file_content = await file.read()
                # Extract embeddings based on file type
                if file.content_type in ["image/png", "image/jpeg"]:
                    text_embedding, img_embedding, raw_img = self._generate_embeddings(text=metadata[i], image=file_content)
                    data.append({
                        "text": metadata[i],
                        "text_embedding": text_embedding,
                        "image_embedding": img_embedding,
                        "image_data": raw_img,
                        "metadata": metadata[i],
                    })

                else:
                    # Chunk and summarize text from PDF
                    try:
                        chunk_text, chunk_metadata = chunk_document(file_content, max_size=200, chunk_overlap=20)
                        text_content = contextualize_chunks(chunk_text, os.getenv('GEMINI_API_KEY'))
                    except Exception as e:
                        raise Exception(f"Failed to extract PDF text: {e}")

                    for text in text_content:
                        text_embedding, _, _ = self._generate_embeddings(text=text)
                        data.append({
                            "text": text,
                            "text_embedding": text_embedding,
                            "image_embedding": None,
                            "image_data": None,
                            "metadata": metadata[i],
                        })

            response = self.db.insert(collection_name, data)
            return response

        except Exception as e:
            raise Exception(f"Failed to upload to user collection: {e}")

    # --------------------------------------------
    # Search utilities
    # --------------------------------------------
    def _generate_embeddings(self, text: str = "", image: bytes = None):
        """
        Generates embeddings for text, image.
        Returns text embedding, image embedding and raw image bytes.
        """
        text_embedding = self.embedding_model_manager.embed_text(text)
        image_embedding, image_raw = None, None
        if image:
            image_embedding, image_raw = self.embedding_model_manager.embed_image(image)
        return text_embedding, image_embedding, image_raw
    
    def _hybrid_search(self, table, text: str, vector_column_name: str, embedding: list, threshold: float) -> pa.Table:
        if not text:
            text = " "

        temp = table.search(query_type="hybrid", 
                         vector_column_name=vector_column_name, 
                         fts_columns="text")\
                 .text(text)\
                 .vector(embedding)\
                 .limit(3)
        
        if len(temp.to_list()) == 0: 
            return pa.Table.from_pydict({})
        
        results = temp.rerank(self.reranker)
        return results.to_arrow().filter(pa.compute.field('_relevance_score') >= threshold)
    
    # --------------------------------------------
    # Search individual collections
    # --------------------------------------------
    def search_multimodal(self, text: str, image: bytes | None) -> pa.Table:
        """
        Perform search with text and image in multimodal (image_text) table.

        Args:
            text (str): text to search
            image (bytes): image to search

        Returns:
            pa.Table: PyArrow table of search result.
        """
        try:
            text_embedding, image_embedding, _ = self._generate_embeddings(text, image)
            if image_embedding:
                return self._hybrid_search(
                    self.image_text_table, text, "image_embedding", image_embedding, THRESHOLD['MULTIMODAL']
                )
            return self._hybrid_search(
                self.image_text_table, text, "text_embedding", text_embedding, THRESHOLD['MULTIMODAL']
            )
        except Exception as e:
            raise Exception(f"Failed searching multimodal: {e}")

    def search_documents(self, text: str) -> pa.Table:
        """
        Perform search with text in document table.
        Hybrid search with vector search on text embedding and full text search on text.

        Args:
            text (str): text to search

        Returns:
            pa.Table: PyArrow table of search result.
        """
        try:
            text_embedding = self.embedding_model_manager.embed_text(text)
            return self._hybrid_search(
                self.doc_table, text, "text_embedding", text_embedding, THRESHOLD['DOCUMENT']
            )
        except Exception as e:
            raise Exception(f"Failed searching document: {e}")

    def search_user(self, text: str, image: bytes | None) -> pa.Table:
        """
        Perform search with text and image on user uploaded collection table.

        Args:
            text (str): text to search
            image (bytes | None): image to search

        Returns:
            pa.Table: PyArrow table of search result.
        """
        try:
            text_embedding, image_embedding, _ = self._generate_embeddings(text, image)
            if image_embedding:
                return self._hybrid_search(
                    self.user_table, text, "image_embedding", image_embedding, THRESHOLD['USER']
                )
            return self._hybrid_search(
                self.user_table, text, "text_embedding", text_embedding, THRESHOLD['USER']
            )
        except Exception as e:
            raise Exception(f"Failed searching user: {e}")

    # --------------------------------------------
    # Search function (Single file)
    # --------------------------------------------
    def search_image_file(self, text: str, image: bytes | None) -> List:
        """
        Performs search for text and image input on user, multimodal and document table.
        Results are combined, and image data is returned as base64 encoded string.

        Args:
            text (str): text to search
            image (bytes | None): image to search
        """
        try:
            u = self.search_user(text, image)
            m = self.search_multimodal(text, image)
            d = self.search_documents(text)

            res = combine_results(user=u, multimodal=m, docs=d)
            res = res.to_pylist()
            for data in res:
                if 'image_data' in data:
                    if data['image_data'] is not None:
                        data['image_data'] = base64.b64encode(
                            data['image_data']).decode("utf-8")

            return res
        except Exception as e:
            raise Exception(f"Error searching for single image file: {e}")
        
    # --------------------------------------------
    # Main search functions
    # --------------------------------------------
    async def search_image(self, files: List[UploadFile], query: str, embedding_model: str, domain: str):
        try:
            self.switch_domain(embedding_model, domain)
            file_content = None
            if not files:
                result = self.search_image_file(text=query, image=file_content)
                return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'success': True, 'content': result}
            
            result = []
            for file in files:
                file_content = await file.read()
                result.extend(self.search_image_file(text=query, image=file_content))

            result = deduplicate(result).to_pylist()
            return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'success': True, 'content': result}
        except Exception as e:
            raise Exception(f"Failed to search with images: {e}")

    async def search_video(self, files: List[UploadFile], query: str, embedding_model: str, domain: str):
        try:
            self.switch_domain(embedding_model, domain)

            d = self.search_documents(text=query)
            u = self.search_user(text=query,image=None)
            m = pa.Table.from_pydict({})

            result = combine_results(user=u, multimodal=m, docs=d)
            if "image_embedding" in result.column_names:
                result = result.drop_columns(["image_data", "image_embedding"])
            result = result.to_pylist()
            return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'success': True, 'content': result}
        except Exception as e:
            raise Exception(f"Failed to search with videos: {e}")
        