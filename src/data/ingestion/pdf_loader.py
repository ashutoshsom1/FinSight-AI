"""PDF loader for financial documents with enhanced table and text extraction."""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import PyPDF2
import structlog
from datetime import datetime

from .base_loader import BaseDocumentLoader
from ..models.document import DocumentChunk, SectionType, DocumentMetadata

logger = structlog.get_logger()


class PDFLoader(BaseDocumentLoader):
    """Loader for PDF financial documents."""
    
    def __init__(self, metadata: DocumentMetadata, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(metadata)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def load(self) -> List[DocumentChunk]:
        """Load and parse PDF into chunks."""
        self.logger.info("Starting PDF loading", file_path=self.metadata.file_path)
        
        try:
            # Extract raw text
            raw_text = self.extract_text(Path(self.metadata.file_path))
            
            # Clean the text
            cleaned_text = self.clean_text(raw_text)
            
            # Create chunks
            chunks = await self._create_chunks(cleaned_text)
            
            # Validate and filter chunks
            valid_chunks = [chunk for chunk in chunks if self.validate_chunk(chunk)]
            
            self.logger.info(
                "PDF loading completed",
                total_chunks=len(chunks),
                valid_chunks=len(valid_chunks)
            )
            
            return valid_chunks
            
        except Exception as e:
            self.logger.error("Failed to load PDF", error=str(e))
            raise
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Update metadata with page count
                self.metadata.page_count = len(pdf_reader.pages)
                
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Add page marker for chunk metadata
                            text_content.append(f"[PAGE {page_num}]\n{page_text}")
                    except Exception as e:
                        self.logger.warning(
                            "Failed to extract text from page",
                            page_num=page_num,
                            error=str(e)
                        )
                        continue
                
                return '\n\n'.join(text_content)
                
        except Exception as e:
            self.logger.error("Failed to extract text from PDF", error=str(e))
            raise
    
    async def _create_chunks(self, text: str) -> List[DocumentChunk]:
        """Create semantic chunks from cleaned text."""
        chunks = []
        
        # Split by page markers first
        pages = text.split('[PAGE ')
        
        current_chunk_text = ""
        current_page = 1
        chunk_index = 0
        
        for page_content in pages[1:]:  # Skip first empty split
            # Extract page number
            lines = page_content.split('\n', 1)
            if len(lines) < 2:
                continue
                
            try:
                page_num = int(lines[0].split(']')[0])
                page_text = lines[1] if len(lines) > 1 else ""
            except (ValueError, IndexError):
                page_num = current_page
                page_text = page_content
            
            # Add to current chunk
            current_chunk_text += f"\n{page_text}"
            
            # Check if we should create a chunk
            if len(current_chunk_text) >= self.chunk_size:
                chunk = await self._create_chunk(
                    current_chunk_text,
                    chunk_index,
                    page_num
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk_text[-self.chunk_overlap:]
                    current_chunk_text = overlap_text
                else:
                    current_chunk_text = ""
            
            current_page = page_num
        
        # Handle remaining text
        if current_chunk_text.strip():
            chunk = await self._create_chunk(
                current_chunk_text,
                chunk_index,
                current_page
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    async def _create_chunk(
        self, 
        text: str, 
        chunk_index: int, 
        page_number: int
    ) -> Optional[DocumentChunk]:
        """Create a single document chunk with metadata."""
        
        cleaned_text = text.strip()
        if not cleaned_text:
            return None
        
        # Detect section type
        section_type = self.detect_section_type(cleaned_text, page_number)
        
        # Extract financial metrics
        financial_metrics = self.extract_financial_metrics(cleaned_text)
        
        # Count tokens (approximate)
        token_count = len(cleaned_text.split()) * 1.3  # Rough approximation
        
        chunk = DocumentChunk(
            document_id=self.metadata.document_id,
            content=cleaned_text,
            section_type=section_type,
            page_number=page_number,
            chunk_index=chunk_index,
            token_count=int(token_count),
            contains_tables=self.contains_tables(cleaned_text),
            contains_numbers=self.contains_numbers(cleaned_text),
            financial_metrics=financial_metrics
        )
        
        return chunk
    
    def extract_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF (placeholder for advanced table extraction)."""
        # This is a placeholder for more sophisticated table extraction
        # Could use libraries like camelot-py, tabula-py, or pdfplumber
        self.logger.info("Table extraction not yet implemented")
        return []
    
    async def extract_metadata_from_content(self, text: str) -> Dict[str, Any]:
        """Extract additional metadata from document content."""
        metadata_updates = {}
        
        # Try to extract filing date
        import re
        date_patterns = [
            r'filed?\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'dated?\s+(\w+\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    from dateutil.parser import parse
                    filing_date = parse(match.group(1))
                    metadata_updates['filing_date'] = filing_date
                    break
                except:
                    continue
        
        # Extract currency information
        currency_patterns = {
            'USD': [r'\$', r'US\s*\$', r'USD', r'dollars?'],
            'GBP': [r'£', r'GBP', r'pounds?', r'sterling'],
            'INR': [r'₹', r'INR', r'rupees?'],
            'EUR': [r'€', r'EUR', r'euros?']
        }
        
        for currency, patterns in currency_patterns.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                metadata_updates['currency'] = currency
                break
        
        return metadata_updates
