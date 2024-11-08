from langchain.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading of different document types using LangChain's document loaders"""

    def load_document(self, file_path: str):
        """Load a single document"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def load_directory(self, dir_path: str):
        """Load all documents from a directory"""
        try:
            loader = DirectoryLoader(dir_path)
            documents = loader.load()
            return documents
        except Exception as e:
            logger.error(f"Error loading directory {dir_path}: {str(e)}")
            raise
