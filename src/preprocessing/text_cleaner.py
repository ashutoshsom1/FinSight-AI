"""Advanced text cleaning for financial documents."""

import re
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger()


class FinancialTextCleaner:
    """Specialized text cleaner for financial documents."""
    
    def __init__(self):
        self.logger = logger.bind(component="text_cleaner")
        
        # Common patterns to remove or standardize
        self.noise_patterns = [
            # Page numbers and references
            (r'Page \d+(?:\s+of \s+\d+)?', ''),
            (r'\d+\s*\|\s*Page', ''),
            (r'^\s*\d+\s*$', ''),  # Standalone page numbers
            
            # Headers and footers
            (r'Annual Report \d{4}', ''),
            (r'Quarterly Report.*?\d{4}', ''),
            (r'Confidential.*?(?=\n)', ''),
            (r'Internal Use Only.*?(?=\n)', ''),
            (r'©.*?\d{4}.*?(?=\n)', ''),
            
            # Legal disclaimers (preserve but standardize)
            (r'This document contains forward-looking statements.*?(?=\n\n)', '[FORWARD_LOOKING_DISCLAIMER]'),
            
            # OCR artifacts
            (r'[^\w\s\.,;:!?()[\]{}"\'-]', ' '),  # Non-standard characters
            (r'\s+', ' '),  # Multiple spaces
            (r'\n\s*\n\s*\n+', '\n\n'),  # Multiple newlines
        ]
        
        # Financial number standardization patterns
        self.number_patterns = [
            # Standardize currency representations
            (r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', r'USD \1'),
            (r'£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', r'GBP \1'),
            (r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', r'INR \1'),
            (r'€\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', r'EUR \1'),
            
            # Standardize large number representations
            (r'(\d+(?:,\d{3})*)\s*million', r'\1M'),
            (r'(\d+(?:,\d{3})*)\s*billion', r'\1B'),
            (r'(\d+(?:,\d{3})*)\s*thousand', r'\1K'),
        ]
    
    def clean_text(self, text: str, preserve_financial_formatting: bool = True) -> str:
        """Clean text while preserving important financial information."""
        
        self.logger.debug("Starting text cleaning", text_length=len(text))
        
        cleaned_text = text
        
        # Apply noise removal patterns
        for pattern, replacement in self.noise_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Standardize financial numbers if requested
        if preserve_financial_formatting:
            for pattern, replacement in self.number_patterns:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        cleaned_text = self._normalize_whitespace(cleaned_text)
        
        # Remove empty lines and excessive spacing
        cleaned_text = self._remove_excessive_spacing(cleaned_text)
        
        self.logger.debug(
            "Text cleaning completed", 
            original_length=len(text),
            cleaned_length=len(cleaned_text)
        )
        
        return cleaned_text.strip()
    
    def extract_tables_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract and structure table-like content from text."""
        tables = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Look for table patterns
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect table start (headers with separators)
            if self._is_table_header(line) and not in_table:
                in_table = True
                current_table = [line]
                continue
            
            # Detect table continuation
            if in_table:
                if self._is_table_row(line):
                    current_table.append(line)
                elif line == '' and len(current_table) > 1:
                    # End of table
                    tables.append({
                        'content': '\n'.join(current_table),
                        'start_line': i - len(current_table),
                        'end_line': i,
                        'row_count': len(current_table) - 1  # Exclude header
                    })
                    current_table = []
                    in_table = False
                elif not line:
                    continue  # Skip empty lines within table
                else:
                    # Non-table content, end current table
                    if len(current_table) > 1:
                        tables.append({
                            'content': '\n'.join(current_table),
                            'start_line': i - len(current_table),
                            'end_line': i,
                            'row_count': len(current_table) - 1
                        })
                    current_table = []
                    in_table = False
        
        # Handle table at end of text
        if in_table and len(current_table) > 1:
            tables.append({
                'content': '\n'.join(current_table),
                'start_line': len(lines) - len(current_table),
                'end_line': len(lines),
                'row_count': len(current_table) - 1
            })
        
        self.logger.debug("Table extraction completed", table_count=len(tables))
        return tables
    
    def _is_table_header(self, line: str) -> bool:
        """Check if line looks like a table header."""
        # Look for multiple words separated by spaces/tabs
        words = line.split()
        if len(words) < 2:
            return False
        
        # Check for financial terms in headers
        financial_terms = [
            'year', 'quarter', 'revenue', 'income', 'profit', 'assets', 
            'equity', 'debt', 'cash', 'margin', 'ratio', 'growth',
            '2020', '2021', '2022', '2023', '2024'
        ]
        
        return any(term.lower() in line.lower() for term in financial_terms)
    
    def _is_table_row(self, line: str) -> bool:
        """Check if line looks like a table row."""
        if not line.strip():
            return False
        
        # Look for numbers and separators
        has_numbers = bool(re.search(r'\d+', line))
        has_separators = bool(re.search(r'\s{2,}|\t', line))
        
        return has_numbers and (has_separators or len(line.split()) > 2)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace tabs with spaces
        text = text.replace('\t', '  ')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing spaces from lines
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        
        return '\n'.join(lines)
    
    def _remove_excessive_spacing(self, text: str) -> str:
        """Remove excessive spacing while preserving paragraph structure."""
        # Remove more than 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at the beginning and end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        
        # Remove completely empty lines between content
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if line.strip() == '':
                if not prev_empty:
                    cleaned_lines.append('')
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        return '\n'.join(cleaned_lines)
    
    def standardize_financial_terms(self, text: str) -> str:
        """Standardize financial terminology across documents."""
        from config.settings import FINANCIAL_METRICS_MAPPING
        
        standardized_text = text
        
        # Create reverse mapping for standardization
        for standard_term, variations in FINANCIAL_METRICS_MAPPING.items():
            for variation in variations:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variation) + r'\b'
                replacement = standard_term.replace('_', ' ').title()
                standardized_text = re.sub(
                    pattern, 
                    replacement, 
                    standardized_text, 
                    flags=re.IGNORECASE
                )
        
        return standardized_text
