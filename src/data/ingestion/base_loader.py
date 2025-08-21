"""Base loader interface for financial documents."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

from ..models.document import DocumentMetadata, DocumentChunk, SectionType

logger = structlog.get_logger()


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    def __init__(self, metadata: DocumentMetadata):
        self.metadata = metadata
        self.logger = logger.bind(
            document_id=metadata.document_id,
            company=metadata.company,
            year=metadata.year
        )
    
    @abstractmethod
    async def load(self) -> List[DocumentChunk]:
        """Load and parse the document into chunks."""
        pass
    
    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Extract raw text from the document."""
        pass
    
    def detect_section_type(self, text: str, page_num: Optional[int] = None) -> SectionType:
        """Detect the section type based on text content and context."""
        text_lower = text.lower()
        
        # Define section keywords
        section_keywords = {
            SectionType.EXECUTIVE_SUMMARY: [
                "executive summary", "highlights", "key highlights", "overview"
            ],
            SectionType.MD_A: [
                "management discussion", "md&a", "management's discussion",
                "analysis of financial condition", "results of operations"
            ],
            SectionType.FINANCIAL_STATEMENTS: [
                "financial statements", "consolidated statements"
            ],
            SectionType.INCOME_STATEMENT: [
                "income statement", "profit and loss", "statement of earnings",
                "consolidated income", "statement of operations"
            ],
            SectionType.BALANCE_SHEET: [
                "balance sheet", "statement of financial position",
                "consolidated balance sheet"
            ],
            SectionType.CASH_FLOW: [
                "cash flow", "statement of cash flows", "consolidated cash flow"
            ],
            SectionType.NOTES: [
                "notes to", "notes on", "financial statements notes",
                "accounting policies", "significant accounting"
            ],
            SectionType.RISK_FACTORS: [
                "risk factors", "principal risks", "risk management",
                "risks and uncertainties"
            ],
            SectionType.BUSINESS_OVERVIEW: [
                "business overview", "our business", "business description",
                "operations", "business segments"
            ],
            SectionType.GOVERNANCE: [
                "corporate governance", "governance", "board of directors",
                "directors' report"
            ],
            SectionType.AUDITOR_REPORT: [
                "auditor", "independent auditor", "audit report",
                "auditors' report"
            ]
        }
        
        # Check for section keywords
        for section_type, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return section_type
        
        return SectionType.OTHER
    
    def extract_financial_metrics(self, text: str) -> List[str]:
        """Extract potential financial metrics from text."""
        from config.settings import FINANCIAL_METRICS_MAPPING
        
        text_lower = text.lower()
        found_metrics = []
        
        for standard_name, variations in FINANCIAL_METRICS_MAPPING.items():
            if any(variation in text_lower for variation in variations):
                found_metrics.append(standard_name)
        
        return found_metrics
    
    def contains_tables(self, text: str) -> bool:
        """Check if text likely contains tabular data."""
        # Simple heuristics for table detection
        lines = text.split('\n')
        
        # Look for patterns indicating tables
        table_indicators = 0
        for line in lines:
            # Multiple numbers in a line
            if len([char for char in line if char.isdigit()]) > 5:
                table_indicators += 1
            # Multiple spaces/tabs (column separation)
            if '  ' in line or '\t' in line:
                table_indicators += 1
            # Currency symbols
            if any(symbol in line for symbol in ['$', '£', '€', '₹', 'USD', 'GBP', 'INR']):
                table_indicators += 1
        
        return table_indicators > len(lines) * 0.3  # 30% of lines show table patterns
    
    def contains_numbers(self, text: str) -> bool:
        """Check if text contains numerical data."""
        import re
        # Look for numbers with potential financial formatting
        number_patterns = [
            r'\d+[,.]?\d*',  # Basic numbers
            r'\(\d+[,.]?\d*\)',  # Numbers in parentheses (negative)
            r'\d+[,.]?\d*[KMB]',  # Numbers with K/M/B suffixes
            r'[£$€₹]\s*\d+',  # Currency symbols
        ]
        
        for pattern in number_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing headers, footers, and noise."""
        import re
        
        # Remove page numbers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+ \| Page', '', text, flags=re.IGNORECASE)
        
        # Remove common headers/footers
        text = re.sub(r'Annual Report \d{4}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Confidential.*?(?=\n)', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines
        text = re.sub(r' +', ' ', text)  # Multiple spaces
        
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\.,;:!?()[\]{}"\'-]', ' ', text)
        
        return text.strip()
    
    def validate_chunk(self, chunk: DocumentChunk) -> bool:
        """Validate that a chunk meets quality criteria."""
        # Minimum content length
        if len(chunk.content.strip()) < 50:
            return False
        
        # Not just numbers or special characters
        alpha_chars = sum(1 for char in chunk.content if char.isalpha())
        if alpha_chars < len(chunk.content) * 0.3:
            return False
        
        # Not repetitive content (common in headers/footers)
        words = chunk.content.lower().split()
        if len(set(words)) < len(words) * 0.5 and len(words) > 10:
            return False
        
        return True
