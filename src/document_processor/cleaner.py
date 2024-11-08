# src/document_processor/cleaner.py
import re
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """Cleans text by removing unwanted characters and whitespace"""

    def clean_text(self, text: str) -> str:
        try:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            raise
