"""
Unit tests for DoclingParser
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Mock docling if not installed
sys.modules['docling'] = MagicMock()
sys.modules['docling.document_converter'] = MagicMock()
sys.modules['docling.datamodel.base_models'] = MagicMock()
sys.modules['docling.datamodel.document'] = MagicMock()

from src.preprocessing.parsers.docling_parser import DoclingParser
from src.preprocessing.parser_base import TextEntry, SourceMetadata

class TestDoclingParser(unittest.TestCase):
    
    def setUp(self):
        # Setup mocks
        self.mock_converter = MagicMock()
        self.mock_doc = MagicMock()
        self.mock_converter.convert.return_value.document = self.mock_doc
        
        # Patch DocumentConverter in the module
        with patch('src.preprocessing.parsers.docling_parser.DocumentConverter', return_value=self.mock_converter):
            self.parser = DoclingParser()

    def test_supported_extensions(self):
        extensions = self.parser.supported_extensions
        self.assertIn('.pdf', extensions)
        self.assertIn('.docx', extensions)
        self.assertIn('.md', extensions)
        self.assertIn('.html', extensions)

    @patch('src.preprocessing.parsers.docling_parser.Path.exists')
    @patch('src.preprocessing.parsers.docling_parser.Path.stat')
    def test_parse_success(self, mock_stat, mock_exists):
        # Setup file path mock
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.name = "test_doc.pdf"
        mock_path.stem = "test_doc"
        mock_path.suffix = ".pdf"
        mock_path.absolute.return_value = "/abs/path/to/test_doc.pdf"
        
        # Setup stat mock
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_mtime = 1600000000
        mock_stat.return_value = mock_stat_obj
        
        # Setup docling document content
        mock_text_item1 = MagicMock()
        mock_text_item1.text = "Hello World"
        mock_text_item1.label = "title"
        mock_text_item1.prov = [MagicMock(page_no=1)]
        
        mock_text_item2 = MagicMock()
        mock_text_item2.text = "This is a paragraph."
        mock_text_item2.label = "paragraph"
        mock_text_item2.prov = [MagicMock(page_no=1)]
        
        self.mock_doc.texts.return_value = [mock_text_item1, mock_text_item2]
        self.mock_doc.name = "test_doc"
        self.mock_doc.pages = [1]
        
        # Run parse
        entries, metadata = self.parser.parse(mock_path)
        
        # Verify results
        self.assertEqual(len(entries), 2)
        self.assertIsInstance(entries[0], TextEntry)
        self.assertEqual(entries[0].text, "Hello World")
        self.assertEqual(entries[0].extra['type'], "title")
        self.assertEqual(entries[0].extra['page'], 1)
        
        self.assertEqual(entries[1].text, "This is a paragraph.")
        
        self.assertIsInstance(metadata, SourceMetadata)
        self.assertEqual(metadata.title, "test_doc")
        self.assertEqual(metadata.source_type, "pdf")
        self.assertEqual(metadata.extra['docling_name'], "test_doc")

    def test_parse_file_not_found(self):
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            self.parser.parse(mock_path)

    @patch('src.preprocessing.parsers.docling_parser.Path.exists')
    @patch('src.preprocessing.parsers.docling_parser.Path.stat')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data="1\n00:00:01,000 --> 00:00:04,000\nHello SRT\n\n2\n00:00:05,000 --> 00:00:08,000\nWorld")
    def test_parse_srt_fallback(self, mock_open, mock_stat, mock_exists):
        # Setup file path mock
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.name = "test.srt"
        mock_path.stem = "test"
        mock_path.suffix = ".srt"
        mock_path.absolute.return_value = "/abs/path/to/test.srt"
        
        # Setup stat mock
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_mtime = 1600000000
        mock_stat.return_value = mock_stat_obj

        # Make docling raise an exception
        self.mock_converter.convert.side_effect = Exception("File format not allowed")

        # Run parse
        entries, metadata = self.parser.parse(mock_path)

        # Verify results
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].text, "Hello SRT")
        self.assertEqual(entries[0].extra['timestamps'], "00:00:01,000 --> 00:00:04,000")
        self.assertEqual(entries[1].text, "World")
        self.assertEqual(metadata.source_type, "srt")

if __name__ == '__main__':
    unittest.main()
