# src/embeddings/embedding_manager.py
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
import logging

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Handles document embeddings and similarity search"""

    def __init__(self, bedrock_client, model_id="amazon.titan-embed-text-v1"):
        self.embeddings = BedrockEmbeddings(
            client=bedrock_client,
            model_id=model_id
        )
        self.vector_store = None

    def create_embeddings(self, documents):
        """Create embeddings for documents and store in FAISS"""
        try:
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            logger.info("Embeddings created and stored in FAISS")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def find_relevant_chunks(self, query: str, k: int = 3):
        """Find most relevant document chunks for a query"""
        if not self.vector_store:
            raise ValueError("No documents have been embedded yet")
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
