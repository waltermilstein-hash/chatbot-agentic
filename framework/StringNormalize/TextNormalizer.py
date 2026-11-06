"""
Text normalizer.
"""
import unicodedata
import re
from typing import Optional


class TextNormalizer:
    """
    Class to normalize text by removing accents, ñ and special characters.
    Useful for searches in PostgreSQL and other systems that do not support accents.
    """

    # Specific replacements for Spanish and other languages
    REPLACEMENTS = {
        'ñ': 'n',
        'ç': 'c',
        'ü': 'u',
        'ï': 'i',
        'ä': 'a',
        'ö': 'o',
        'ë': 'e',
    }

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text by removing accents, ñ and special characters.

        Args:
            text: Text to normalize

        Returns:
            Normalized text without accents or special characters

        Examples:
            >>> TextNormalizer.normalize("José María")
            "jose maria"
            >>> TextNormalizer.normalize("Niño español")
            "nino espanol"
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Specific replacements for Spanish
        for orig, repl in TextNormalizer.REPLACEMENTS.items():
            text = text.replace(orig, repl)

        # Decompose Unicode characters (separates letter from accent)
        # NFD = Canonical Decomposition
        text = unicodedata.normalize('NFD', text)

        # Keep only ASCII characters (removes the accents)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

        # Recompose to NFC (canonical composed form)
        text = unicodedata.normalize('NFC', text)

        # Remove special characters (leaving only letters, numbers and spaces)
        text = re.sub(r'[^a-z0-9\s]', '', text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def normalize_aggressive(text: str) -> str:
        """
        Aggressive normalization for more flexible searches.
        Removes everything except letters and numbers (no spaces).

        Args:
            text: Text to normalize

        Returns:
            Normalized text with only letters and numbers

        Examples:
            >>> TextNormalizer.normalize_aggressive("José María 123!")
            "josemaria123"
        """
        if not text:
            return ""

        # Apply basic normalization
        text = TextNormalizer.normalize(text)

        # Remove EVERYTHING except letters and numbers (no spaces)
        text = re.sub(r'[^a-z0-9]', '', text)

        return text

    @staticmethod
    def search_contains(query: str, content: str, aggressive: bool = False) -> bool:
        """
        Compare normalized query and content to check whether query is in content.

        Args:
            query: Search text
            content: Content to search in
            aggressive: If True, use aggressive normalization

        Returns:
            True if the normalized query is in the normalized content

        Examples:
            >>> TextNormalizer.search_contains("José", "Jose Maria")
            True
            >>> TextNormalizer.search_contains("niño", "El nino juega")
            True
        """
        if not query or not content:
            return False

        normalize_func = (TextNormalizer.normalize_aggressive if aggressive
                          else TextNormalizer.normalize)

        normalized_query = normalize_func(query)
        normalized_content = normalize_func(content)

        return normalized_query in normalized_content

    @classmethod
    def batch_normalize(cls, texts: list[str], aggressive: bool = False) -> list[str]:
        """
        Normalize a list of texts.

        Args:
            texts: List of texts to normalize
            aggressive: If True, use aggressive normalization

        Returns:
            List of normalized texts

        Examples:
            >>> TextNormalizer.batch_normalize(["José", "María", "Niño"])
            ["jose", "maria", "nino"]
        """
        normalize_func = cls.normalize_aggressive if aggressive else cls.normalize
        return [normalize_func(text) for text in texts if text]

    @staticmethod
    def normalize_safe(text: Optional[str], default: str = "") -> str:
        """
        Safe version of normalize that handles None and exceptions.

        Args:
            text: Text to normalize (may be None)
            default: Default value if text is None or an error occurs

        Returns:
            Normalized text or the default value
        """
        try:
            if text is None:
                return default
            return TextNormalizer.normalize(text)
        except Exception:
            return default


# Alias for compatibility with legacy code
def normalize_text(text: str) -> str:
    """Helper function kept for backwards compatibility."""
    return TextNormalizer.normalize(text)


def normalize_text_aggressive(text: str) -> str:
    """Helper function kept for backwards compatibility."""
    return TextNormalizer.normalize_aggressive(text)


def search_normalize(query: str, content: str) -> bool:
    """Helper function kept for backwards compatibility."""
    return TextNormalizer.search_contains(query, content)
