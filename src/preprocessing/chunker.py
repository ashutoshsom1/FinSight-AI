"""Semantic chunking for financial documents."""

from typing import List, Dict, Any, Optional
import structlog
import tiktoken
from dataclasses import dataclass

from ..data.models.document import DocumentChunk, SectionType

logger = structlog.get_logger()


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 1000  # Target tokens per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum viable chunk size
    max_chunk_size: int = 2000  # Maximum chunk size
    preserve_sentences: bool = True  # Try to break at sentence boundaries
    preserve_paragraphs: bool = True  # Try to break at paragraph boundaries


class SemanticChunker:
    """Advanced chunker that respects document structure and semantics."""
    
    def __init__(self, config: ChunkingConfig, encoding_name: str = "cl100k_base"):
        self.config = config
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.logger = logger.bind(component="semantic_chunker")
    
    async def chunk_document(
        self, 
        text: str, 
        document_id: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Chunk a document into semantic segments."""
        
        self.logger.info("Starting semantic chunking", document_id=document_id)
        
        # First, split by major sections if possible
        sections = self._split_by_sections(text)
        
        all_chunks = []
        chunk_index = 0
        
        for section_text, section_type in sections:
            # Chunk each section separately
            section_chunks = await self._chunk_section(
                section_text, 
                document_id, 
                section_type,
                chunk_index
            )
            
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        self.logger.info(
            "Semantic chunking completed", 
            document_id=document_id,
            total_chunks=len(all_chunks)
        )
        
        return all_chunks
    
    def _split_by_sections(self, text: str) -> List[tuple[str, SectionType]]:
        """Split document into major sections."""
        
        # Define section headers to look for
        section_headers = {
            SectionType.EXECUTIVE_SUMMARY: [
                r'executive\s+summary', r'key\s+highlights', r'highlights',
                r'overview', r'summary'
            ],
            SectionType.MD_A: [
                r'management.?s?\s+discussion', r'md&a', 
                r'analysis\s+of\s+financial\s+condition',
                r'results\s+of\s+operations'
            ],
            SectionType.FINANCIAL_STATEMENTS: [
                r'financial\s+statements', r'consolidated\s+statements'
            ],
            SectionType.INCOME_STATEMENT: [
                r'income\s+statement', r'profit\s+and\s+loss',
                r'statement\s+of\s+earnings', r'statement\s+of\s+operations'
            ],
            SectionType.BALANCE_SHEET: [
                r'balance\s+sheet', r'statement\s+of\s+financial\s+position'
            ],
            SectionType.CASH_FLOW: [
                r'cash\s+flow', r'statement\s+of\s+cash\s+flows'
            ],
            SectionType.NOTES: [
                r'notes\s+to.*?financial\s+statements', r'notes\s+on\s+accounts',
                r'accounting\s+policies', r'significant\s+accounting'
            ],
            SectionType.RISK_FACTORS: [
                r'risk\s+factors', r'principal\s+risks', r'risk\s+management'
            ],
            SectionType.BUSINESS_OVERVIEW: [
                r'business\s+overview', r'our\s+business', r'operations',
                r'business\s+segments'
            ]
        }
        
        sections = []
        current_section = SectionType.OTHER
        current_text = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line is a section header
            detected_section = None
            for section_type, patterns in section_headers.items():
                for pattern in patterns:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        detected_section = section_type
                        break
                if detected_section:
                    break
            
            # If we found a new section header
            if detected_section and detected_section != current_section:
                # Save previous section if it has content
                if current_text.strip():
                    sections.append((current_text.strip(), current_section))
                
                # Start new section
                current_section = detected_section
                current_text = line + '\n'
            else:
                # Add to current section
                current_text += line + '\n'
        
        # Add final section
        if current_text.strip():
            sections.append((current_text.strip(), current_section))
        
        # If no sections were detected, treat entire text as one section
        if not sections:
            sections = [(text, SectionType.OTHER)]
        
        self.logger.debug("Section splitting completed", section_count=len(sections))
        return sections
    
    async def _chunk_section(
        self, 
        text: str, 
        document_id: str, 
        section_type: SectionType,
        start_chunk_index: int
    ) -> List[DocumentChunk]:
        """Chunk a single section into smaller pieces."""
        
        chunks = []
        
        # If section is small enough, keep as single chunk
        token_count = self._count_tokens(text)
        if token_count <= self.config.chunk_size:
            chunk = DocumentChunk(
                document_id=document_id,
                content=text,
                section_type=section_type,
                chunk_index=start_chunk_index,
                token_count=token_count,
                contains_tables=self._contains_tables(text),
                contains_numbers=self._contains_numbers(text),
                financial_metrics=self._extract_financial_metrics(text)
            )
            return [chunk]
        
        # Split into smaller chunks
        if self.config.preserve_paragraphs:
            paragraphs = text.split('\n\n')
            chunks = await self._chunk_by_paragraphs(
                paragraphs, document_id, section_type, start_chunk_index
            )
        else:
            chunks = await self._chunk_by_tokens(
                text, document_id, section_type, start_chunk_index
            )
        
        return chunks
    
    async def _chunk_by_paragraphs(
        self, 
        paragraphs: List[str], 
        document_id: str, 
        section_type: SectionType,
        start_chunk_index: int
    ) -> List[DocumentChunk]:
        """Chunk by paragraphs while respecting token limits."""
        
        chunks = []
        current_chunk_text = ""
        current_chunk_paragraphs = []
        chunk_index = start_chunk_index
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            test_text = current_chunk_text + '\n\n' + paragraph if current_chunk_text else paragraph
            test_tokens = self._count_tokens(test_text)
            
            if test_tokens <= self.config.chunk_size or not current_chunk_text:
                # Add paragraph to current chunk
                current_chunk_paragraphs.append(paragraph)
                current_chunk_text = '\n\n'.join(current_chunk_paragraphs)
            else:
                # Create chunk from current paragraphs
                if current_chunk_text:
                    chunk = await self._create_chunk(
                        current_chunk_text, document_id, section_type, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap if configured
                if self.config.chunk_overlap > 0 and current_chunk_paragraphs:
                    # Take last paragraph(s) for overlap
                    overlap_text = current_chunk_paragraphs[-1]
                    overlap_tokens = self._count_tokens(overlap_text)
                    
                    if overlap_tokens <= self.config.chunk_overlap:
                        current_chunk_paragraphs = [overlap_text, paragraph]
                    else:
                        current_chunk_paragraphs = [paragraph]
                else:
                    current_chunk_paragraphs = [paragraph]
                
                current_chunk_text = '\n\n'.join(current_chunk_paragraphs)
        
        # Handle remaining content
        if current_chunk_text:
            chunk = await self._create_chunk(
                current_chunk_text, document_id, section_type, chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_by_tokens(
        self, 
        text: str, 
        document_id: str, 
        section_type: SectionType,
        start_chunk_index: int
    ) -> List[DocumentChunk]:
        """Chunk by token count with sentence boundary preservation."""
        
        chunks = []
        
        # Split into sentences if preserving sentence boundaries
        if self.config.preserve_sentences:
            sentences = self._split_sentences(text)
        else:
            # Split by approximate token boundaries
            words = text.split()
            sentences = [' '.join(words[i:i+50]) for i in range(0, len(words), 50)]
        
        current_chunk_text = ""
        chunk_index = start_chunk_index
        
        for sentence in sentences:
            test_text = current_chunk_text + ' ' + sentence if current_chunk_text else sentence
            test_tokens = self._count_tokens(test_text)
            
            if test_tokens <= self.config.chunk_size or not current_chunk_text:
                current_chunk_text = test_text
            else:
                # Create chunk
                if current_chunk_text:
                    chunk = await self._create_chunk(
                        current_chunk_text, document_id, section_type, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Handle overlap
                if self.config.chunk_overlap > 0:
                    overlap_text = current_chunk_text[-self.config.chunk_overlap:]
                    current_chunk_text = overlap_text + ' ' + sentence
                else:
                    current_chunk_text = sentence
        
        # Handle remaining content
        if current_chunk_text:
            chunk = await self._create_chunk(
                current_chunk_text, document_id, section_type, chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _create_chunk(
        self, 
        text: str, 
        document_id: str, 
        section_type: SectionType,
        chunk_index: int
    ) -> DocumentChunk:
        """Create a DocumentChunk with computed metadata."""
        
        return DocumentChunk(
            document_id=document_id,
            content=text.strip(),
            section_type=section_type,
            chunk_index=chunk_index,
            token_count=self._count_tokens(text),
            contains_tables=self._contains_tables(text),
            contains_numbers=self._contains_numbers(text),
            financial_metrics=self._extract_financial_metrics(text)
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be enhanced with NLTK/spaCy)
        import re
        
        # Split on sentence endings, but be careful with abbreviations and numbers
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _contains_tables(self, text: str) -> bool:
        """Check if text contains table-like structures."""
        # Look for multiple aligned numbers or structured data
        lines = text.split('\n')
        table_indicators = 0
        
        for line in lines:
            # Multiple numbers in a line
            if len(re.findall(r'\d+', line)) >= 3:
                table_indicators += 1
            # Multiple spaces (column alignment)
            if re.search(r'\s{3,}', line):
                table_indicators += 1
        
        return table_indicators > len(lines) * 0.2
    
    def _contains_numbers(self, text: str) -> bool:
        """Check if text contains numerical data."""
        return bool(re.search(r'\d+', text))
    
    def _extract_financial_metrics(self, text: str) -> List[str]:
        """Extract financial metrics mentioned in the text."""
        from config.settings import FINANCIAL_METRICS_MAPPING
        
        text_lower = text.lower()
        found_metrics = []
        
        for standard_name, variations in FINANCIAL_METRICS_MAPPING.items():
            if any(variation in text_lower for variation in variations):
                found_metrics.append(standard_name)
        
        return found_metrics
