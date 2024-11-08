# src/document_processor/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Splits documents into chunks"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def chunk_documents(self, documents) -> List[str]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise
