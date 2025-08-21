"""Basic functionality tests for FinSight AI."""

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.models.document import DocumentMetadata, DocumentType, Company, DocumentChunk, SectionType
from preprocessing.text_cleaner import FinancialTextCleaner
from preprocessing.chunker import SemanticChunker, ChunkingConfig


class TestTextCleaner:
    """Test the financial text cleaner."""
    
    def setup_method(self):
        self.cleaner = FinancialTextCleaner()
    
    def test_basic_cleaning(self):
        """Test basic text cleaning functionality."""
        
        dirty_text = """
        Page 15
        Annual Report 2023
        
        Castrol's revenue increased by 12% to $15.2 billion in 2023.
        
        Confidential - Internal Use Only
        
        The company's operating margin improved significantly.
        
        
        
        Page 16
        """
        
        cleaned = self.cleaner.clean_text(dirty_text)
        
        # Check that noise is removed
        assert "Page 15" not in cleaned
        assert "Page 16" not in cleaned
        assert "Annual Report 2023" not in cleaned
        assert "Confidential" not in cleaned
        
        # Check that content is preserved
        assert "Castrol's revenue increased" in cleaned
        assert "operating margin improved" in cleaned
    
    def test_financial_number_standardization(self):
        """Test financial number standardization."""
        
        text = "Revenue was $15.2 million and profit was £8.5 billion."
        cleaned = self.cleaner.clean_text(text, preserve_financial_formatting=True)
        
        # Check standardization
        assert "USD 15.2" in cleaned
        assert "GBP 8.5" in cleaned
    
    def test_table_extraction(self):
        """Test table extraction from text."""
        
        table_text = """
        Financial Summary
        
        Year    Revenue    Profit    Margin
        2022    $14.2B     $1.1B     7.8%
        2023    $15.2B     $1.3B     8.5%
        
        The above table shows strong growth.
        """
        
        tables = self.cleaner.extract_tables_text(table_text)
        
        assert len(tables) > 0
        assert "Revenue" in tables[0]["content"]
        assert tables[0]["row_count"] == 2  # 2 data rows


class TestSemanticChunker:
    """Test the semantic chunker."""
    
    def setup_method(self):
        config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
        self.chunker = SemanticChunker(config)
    
    @pytest.mark.asyncio
    async def test_basic_chunking(self):
        """Test basic document chunking."""
        
        test_text = """
        Executive Summary
        
        Castrol delivered strong financial performance in 2023 with revenue growth of 12%.
        
        Management Discussion and Analysis
        
        Our revenue increased from $13.6 billion in 2022 to $15.2 billion in 2023.
        This growth was driven by strong demand in automotive lubricants.
        
        Financial Statements
        
        Income Statement
        Revenue: $15.2 billion
        Operating Income: $1.3 billion
        Net Income: $950 million
        """
        
        chunks = await self.chunker.chunk_document(test_text, "test_doc_id")
        
        assert len(chunks) > 0
        
        # Check that chunks have proper metadata
        for chunk in chunks:
            assert chunk.document_id == "test_doc_id"
            assert chunk.section_type in SectionType
            assert chunk.chunk_index >= 0
            assert len(chunk.content) > 0
    
    @pytest.mark.asyncio
    async def test_section_detection(self):
        """Test section type detection."""
        
        test_cases = [
            ("Executive Summary\nKey highlights for 2023", SectionType.EXECUTIVE_SUMMARY),
            ("Management Discussion and Analysis\nRevenue analysis", SectionType.MD_A),
            ("Income Statement\nRevenue $15.2B", SectionType.INCOME_STATEMENT),
            ("Balance Sheet\nTotal Assets $25B", SectionType.BALANCE_SHEET),
            ("Risk Factors\nMarket volatility", SectionType.RISK_FACTORS),
        ]
        
        for text, expected_section in test_cases:
            chunks = await self.chunker.chunk_document(text, "test_doc")
            assert len(chunks) > 0
            assert chunks[0].section_type == expected_section


class TestDocumentModels:
    """Test document data models."""
    
    def test_document_metadata_creation(self):
        """Test creating document metadata."""
        
        metadata = DocumentMetadata(
            company=Company.CASTROL,
            document_type=DocumentType.ANNUAL_REPORT,
            year=2023,
            file_path="./data/castrol_2023.pdf"
        )
        
        assert metadata.company == Company.CASTROL
        assert metadata.year == 2023
        assert metadata.document_id is not None
    
    def test_document_chunk_creation(self):
        """Test creating document chunks."""
        
        chunk = DocumentChunk(
            document_id="test_doc",
            content="Castrol's revenue was $15.2 billion in 2023.",
            section_type=SectionType.INCOME_STATEMENT,
            chunk_index=0,
            contains_numbers=True,
            financial_metrics=["revenue"]
        )
        
        assert chunk.document_id == "test_doc"
        assert chunk.contains_numbers is True
        assert "revenue" in chunk.financial_metrics


# Integration test
@pytest.mark.asyncio
async def test_end_to_end_processing():
    """Test end-to-end document processing workflow."""
    
    # This test would require actual services to be initialized
    # For now, it's a placeholder for integration testing
    
    # Sample workflow:
    # 1. Create sample document
    # 2. Process through ingestion pipeline
    # 3. Generate embeddings
    # 4. Store in vector database
    # 5. Retrieve and verify
    
    # This would be implemented once we have sample data
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
