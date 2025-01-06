import base64
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import pyarrow as pa
from db.connection import DBConnection
from db.constants import DOC_COLLECTION_NAME, MM_COLLECTION_NAME, USER_COLLECTION_NAME
from utils.rag_utils import Domain, deduplicate, combine_results
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
        self.image_text_table = None
        self.doc_table = None
        self.user_table = None
        self.domain = Domain.DEFAULT
        self.init_collections()
        self.load_collections()

    def init_collections(self):
        """
        Initialise collections for all domains
        """
        for domain in Domain:
            self.db.create_or_load_collection(
                MM_COLLECTION_NAME, self.embedding_model_manager, domain
            )
            self.db.create_or_load_collection(
                DOC_COLLECTION_NAME, self.embedding_model_manager, domain
            )
            self.db.create_or_load_collection(
                USER_COLLECTION_NAME, self.embedding_model_manager, domain
            )

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
    # Domain related functions
    # --------------------------------------------
    def is_domain_changed(self, domain: str):
        """
        Returns true when domain is different from current domain
        """
        return domain != self.domain.value

    def switch_domain(self, domain: str):
        """
        Update RAG target collection upon domain change
        """
        if self.is_domain_changed(domain):
            self.domain = Domain(domain)
            self.load_collections()

    # --------------------------------------------
    # Reset user collection
    # --------------------------------------------
    def reset_user_table(self, domain: str):
        """
        Reset RAG by dropping all collections of user table for all embedding models.
        """
        self.switch_domain(domain)
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
            self.switch_domain(domain)

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
    

    def _image_vector_search(self, table, embedding: list) -> pa.Table:
        """
        Perform vector search on image embedding via cosine similarity
        """
        try:
            img_vec_search = table.search(embedding, query_type="vector",
                                vector_column_name="image_embedding")\
                                    .metric("cosine")\
                                    .limit(3)
            res = img_vec_search.to_arrow()
            # Compute relevance score based on cosine similarity distance
            distances = np.array(res['_distance'])
            relevance_scores = 1 - distances
            relevance_scores_array = pa.array(relevance_scores)
            res = res.append_column('_relevance_score', relevance_scores_array)
            return res.filter(pa.compute.field('_relevance_score') >= 0.6)
        except Exception as e:
            raise Exception(f"image vector search failed: {e}")
    

    def _full_text_search(self, table, text: str) -> pa.Table:
        """
        Perform full text search on text
        """
        try:
            res = table.search(text, query_type="fts",fts_columns="text")\
                        .limit(5)
            res = res.to_arrow().filter(pa.compute.field('_score') >= 12)
            if res.num_rows > 0:
                # Rename _score to _relevance_score
                new_cols = ['_relevance_score' if name=='_score' else name for name in res.column_names]
                return res.rename_columns(new_cols)
            return res
        except Exception as e:
            raise Exception(f"full text search failed: {e}")


    def _hybrid_search(self, table, text, image_embedding: list = []) -> pa.Table:
        try:
            image_res, text_res = None, None
            if image_embedding: image_res = self._image_vector_search(table, image_embedding)        
            if text:  text_res = self._full_text_search(table, text)

            if image_res and text_res:
                df1 = image_res.to_pandas()
                df2 = text_res.to_pandas()
                merged_df = pd.merge(df1, df2, on=["id", "text"], how="outer", suffixes=("_left", "_right"))
                merged_df["_relevance_score"] = (
                    merged_df["_relevance_score_left"].fillna(0) + 
                    merged_df["_relevance_score_right"].fillna(0)
                )
                merged_df = merged_df.drop(columns=["_score", "_relevance_score_left", "_relevance_score_right"],
                                           errors="ignore")
                final_table = pa.Table.from_pandas(merged_df)
                return final_table
            
            if image_res:   return image_res
            if text_res:    return text_res
            return pa.Table.from_pydict({})
        
        except Exception as e:
            raise Exception(f"hybrid search failed: {e}")

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
            if image:
                image_embedding, _ = self.embedding_model_manager.embed_image(image)
                return self._hybrid_search(self.image_text_table, text, image_embedding)
            return self._full_text_search(self.image_text_table, text)
        
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
            return self._full_text_search(self.doc_table, text)
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
            if image:
                image_embedding, _ = self.embedding_model_manager.embed_image(image)
                return self._hybrid_search(self.user_table, text, image_embedding)
            return self._full_text_search(self.user_table, text)
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
            d = self.search_documents(text) if text else pa.Table.from_pydict({})

            res = combine_results(user=u, multimodal=m, docs=d)
            res = deduplicate(res).to_pylist()
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
            self.switch_domain(domain)
            file_content = None
            if not files:
                if not query:
                    return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'success': True, 'content': []}
                else:
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
            self.switch_domain(domain)

            if not query:
                return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'success': True, 'content': []}
            
            d = self.search_documents(text=query)
            u = self.search_user(text=query,image=None)
            m = pa.Table.from_pydict({})

            result = combine_results(user=u, multimodal=m, docs=d)
            if "image_embedding" in result.column_names:
                result = result.drop_columns(["image_data", "image_embedding"])
            result = deduplicate(result).to_pylist()
            return {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'success': True, 'content': result}
        except Exception as e:
            raise Exception(f"Failed to search with videos: {e}")
        