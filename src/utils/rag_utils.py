import base64
from datetime import datetime
import pyarrow as pa
from lancedb.rerankers import RRFReranker
from db.connection import DBConnection
from db.utils import DOC_COLLECTION_NAME, MM_COLLECTION_NAME, USER_COLLECTION_NAME
from utils.embed_utils import EmbeddingModelManager, EmbeddingModel
from lancedb.table import LanceHybridQueryBuilder

USER_WEIGHT = 0.4
MULTIMODAL_WEIGHT = 0.4
DOCUMENT_WEIGHT = 0.2


class RAGManager():
    def __init__(self):
        self.embedding_model_manager = EmbeddingModelManager()
        self.db = DBConnection()
        self.reranker = RRFReranker()
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

    def search_multimodal(self, text: str, image: bytes | None) -> LanceHybridQueryBuilder:
        """
        Perform search with text and image in multimodal (image_text) table.
        Hybrid search with vector search on image embedding and full text search on text.
        If no image, hybrid search with vector search on text embedding and full text search on text.

        Args:
            text (str): text to search
            image (bytes): image to search

        Returns:
            LanceHybridQueryBuilder: result of search
        """
        try:
            if image:
                search_img_embedding, search_img_raw = self.embedding_model_manager.embed_image(
                    image)
                res = self.image_text_table.search(query_type="hybrid",
                                                   vector_column_name="image_embedding",
                                                   fts_columns="text")\
                    .vector(search_img_embedding)\
                    .text(text)\
                    .limit(3)\
                    .rerank(self.reranker)
            else:
                text_embedding = self.embedding_model_manager.embed_text(text)
                res = self.image_text_table.search(query_type="hybrid",
                                                   vector_column_name="text_embedding",
                                                   fts_columns="text")\
                    .vector(text_embedding)\
                    .text(text)\
                    .limit(3)\
                    .rerank(self.reranker)
            return res
        except Exception as e:
            raise Exception(f"Failed searching multimodal: {e}")

    def search_documents(self, text: str) -> LanceHybridQueryBuilder:
        """
        Perform search with text in document table.
        Hybrid search with vector search on text embedding and full text search on text.

        Args:
            text (str): text to search

        Returns:
            LanceHybridQueryBuilder: result of search
        """
        try:
            search_txt_embedding = self.embedding_model_manager.embed_text(
                text)
            res = self.doc_table.search(query_type="hybrid",
                                        vector_column_name="text_embedding",
                                        fts_columns="text")\
                .vector(search_txt_embedding)\
                .text(text)\
                .select(["text", "metadata"])\
                .limit(3)\
                .rerank(self.reranker)
            return res
        except Exception as e:
            raise Exception(f"Failed searching document: {e}")

    def search_user(self, text: str, image: bytes | None) -> pa.Table:
        """
        Perform search with text and image on user uploaded collection table.
        Hybrid search:
            if user provides text only, then do hybrid search with
            full text search on text, and vector search on text embeddings.

            if user provides text and image, then do hybrid search
            with vector search on text embeddings and vector search on image embeddings.

        Args:
            text (str): text to search
            image (bytes | None): image to search

        Returns:
            pa.Table: PyArrow table of search result.
        """
        try:
            # vector search with text
            search_txt_embedding = self.embedding_model_manager.embed_text(
                text)
            txt_search = self.user_table.search(search_txt_embedding,
                                                vector_column_name="text_embedding")\
                .limit(3)\
                .with_row_id(True)

            if image:
                search_img_embedding, search_img_raw = self.embedding_model_manager.embed_image(
                    image)

                # vector search with image
                img_search = self.user_table.search(search_img_embedding,
                                                    vector_column_name="image_embedding")\
                    .limit(3)\
                    .with_row_id(True)

                # rerank multivector for hybrid search of text, image_embedding, text_embedding
                user_res = self.reranker.rerank_multivector(
                    [img_search, txt_search], text)
                return user_res
            else:
                fts_txt = self.user_table.search(text, query_type="fts")\
                    .limit(3)\
                    .with_row_id(True)\
                    .to_arrow()
                user_res = self.reranker.rerank_hybrid(query=text,
                                                       vector_results=txt_search.to_arrow(),
                                                       fts_results=fts_txt)

            return user_res
        except Exception as e:
            raise Exception(f"Failed searching user: {e}")

    def weighted_table(self, table: pa.Table, weight: float) -> pa.Table:
        """
        Weighs relevance score in table according to weight given.
        Weight is expected to be in float format, i.e. 0.6 for 60%

        Args:
            table (pa.Table): PyArrow table to weight
            weight (float): Weight to apply for relevance score (i.e. 0.6 for 60%)

        Returns:
            pa.Table: PyArrow table with new '_weighted_relevance_score' column.
        """
        try:
            if not isinstance(table, pa.Table):
                table = table.to_arrow()

            if "_rowid" not in table.column_names:
                table = table.append_column(
                    "_rowid", pa.array(range(len(table)), type=pa.uint64())
                )

            table = table.append_column(
                "_weighted_relevance_score",
                pa.array(
                    [weight * score for score in table['_relevance_score'].to_pylist()]
                )
            )
            return table
        except Exception as e:
            raise Exception(f"Failed creating weighted table: {e}")

    def combine_results(self, user: pa.Table,
                        multimodal: LanceHybridQueryBuilder,
                        docs: LanceHybridQueryBuilder,
                        w_user: float = USER_WEIGHT,
                        w_mm: float = MULTIMODAL_WEIGHT,
                        w_doc: float = DOCUMENT_WEIGHT) -> pa.Table:
        """
        Combine the results of searching from user table, document table and 
        multimodal table.
        Apply weightage for each result, and reranks them based on the new
        weighted relevance score in descending order.

        Args:
            user (pa.Table): PyArrow table of user table search result
            multimodal (LanceHybridQueryBuilder): multimodal table search result
            docs (LanceHybridQueryBuilder): document table search result
            w_user (float): weight to apply for user search result
            w_mm (float): weight to apply for multimodal search result
            w_doc (float): weight to apply for document search result

        Returns:
            pa.Table: PyArrow table of sorted combined results.
        """
        try:
            u = self.weighted_table(user, w_user)
            m = self.weighted_table(multimodal, w_mm)
            d = self.weighted_table(docs, w_doc)
            _concat_tables_args = {"promote_options": "default"}
            res = pa.concat_tables([u, m, d], **_concat_tables_args)
            res = res.sort_by(
                [("_weighted_relevance_score", "descending")]
            )
            return res
        except Exception as e:
            raise Exception(f"Failed combining results: {e}")

    def search(self, text: str, image: bytes | None) -> dict:
        """
        Performs search for text and image input on user, multimodal and document table.
        Results are combined, and image data is returned as base64 encoded string.

        Args:
            text (str): text to search
            image (bytes | None): image to search

        Returns:
            Dict: dictionary of 'success' and 'message' containing search result.
        """
        try:
            u = self.search_user(text, image)
            m = self.search_multimodal(text, image)
            d = self.search_documents(text)

            res = self.combine_results(user=u, multimodal=m, docs=d)
            res = res.to_pylist()
            for data in res:
                if 'image_data' in data:
                    if data['image_data'] is not None:
                        data['image_data'] = base64.b64encode(
                            data['image_data']).decode("utf-8")

            return {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "success": True, "message": res}
        except Exception as e:
            raise Exception(f"Error searching: {e}")
