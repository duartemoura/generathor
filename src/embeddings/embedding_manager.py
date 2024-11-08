# src/embeddings/embedding_manager.py
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
import os
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

    def save_embeddings(self, faiss_index_path: str, metadata_path: str = None):
        """Save FAISS index and metadata to disk"""
        if not self.vector_store:
            raise ValueError("No embeddings to save")
        try:
            self.vector_store.save_local(faiss_index_path)
            if metadata_path:
                with open(metadata_path, 'w') as f:
                    f.write("Metadata for embeddings")  # Replace with actual metadata if necessary
            logger.info(f"Embeddings saved to {faiss_index_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise

    def load_embeddings(self, faiss_index_path: str):
        """Load FAISS index from disk"""
        try:
            self.vector_store = FAISS.load_local(faiss_index_path, self.embeddings)
            logger.info(f"Embeddings loaded from {faiss_index_path}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def wipe_embeddings(self, faiss_index_path: str, metadata_path: str = None):
        """Delete the existing FAISS database and metadata files"""
        try:
            if os.path.exists(faiss_index_path):
                for file in os.listdir(faiss_index_path):
                    os.remove(os.path.join(faiss_index_path, file))
                os.rmdir(faiss_index_path)
                logger.info(f"Wiped FAISS database at {faiss_index_path}")
            else:
                logger.warning(f"No FAISS database found at {faiss_index_path} to wipe")
            
            if metadata_path and os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info(f"Wiped metadata file at {metadata_path}")
        except Exception as e:
            logger.error(f"Error wiping embeddings: {str(e)}")
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
