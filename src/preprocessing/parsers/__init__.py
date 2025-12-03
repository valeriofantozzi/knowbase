"""
Document Parsers Package

Contains format-specific document parsers that implement the DocumentParser interface.
"""

from .docling_parser import DoclingParser

__all__ = [
    "DoclingParser",
]
