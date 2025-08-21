"""Data models for financial documents and metadata."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from uuid import uuid4


class DocumentType(str, Enum):
    """Types of financial documents."""
    ANNUAL_REPORT = "annual_report"
    QUARTERLY_REPORT = "quarterly_report"
    PRESS_RELEASE = "press_release"
    SUSTAINABILITY_REPORT = "sustainability_report"
    INVESTOR_PRESENTATION = "investor_presentation"


class SectionType(str, Enum):
    """Types of sections within financial documents."""
    EXECUTIVE_SUMMARY = "executive_summary"
    MD_A = "management_discussion_analysis"  # MD&A
    FINANCIAL_STATEMENTS = "financial_statements"
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow_statement"
    NOTES = "notes_to_financial_statements"
    RISK_FACTORS = "risk_factors"
    BUSINESS_OVERVIEW = "business_overview"
    GOVERNANCE = "corporate_governance"
    AUDITOR_REPORT = "auditor_report"
    OTHER = "other"


class Company(str, Enum):
    """Supported companies."""
    CASTROL = "castrol"
    VEEDOL = "veedol"
    VALVOLINE = "valvoline"


class DocumentMetadata(BaseModel):
    """Metadata for financial documents."""
    
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    company: Company
    document_type: DocumentType
    year: int
    quarter: Optional[int] = None  # For quarterly reports
    filing_date: Optional[datetime] = None
    source_url: Optional[str] = None
    file_path: str
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    language: str = "en"
    currency: Optional[str] = None
    accounting_standard: Optional[str] = None  # IFRS, US GAAP, Ind AS
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    processing_version: str = "1.0"
    extraction_method: Optional[str] = None  # pypdf, ocr, etc.
    
    class Config:
        use_enum_values = True


class DocumentChunk(BaseModel):
    """A chunk of text from a financial document."""
    
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    section_type: SectionType
    page_number: Optional[int] = None
    chunk_index: int  # Order within document
    
    # Chunk-specific metadata
    token_count: Optional[int] = None
    contains_tables: bool = False
    contains_numbers: bool = False
    financial_metrics: List[str] = Field(default_factory=list)
    
    # Vector embedding (stored separately in vector DB)
    embedding_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


class FinancialMetric(BaseModel):
    """Extracted financial metric with standardized naming."""
    
    metric_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_id: str
    
    # Standardized metric information
    metric_name: str  # Standardized name (e.g., "revenue")
    original_name: str  # As it appears in document
    value: Optional[float] = None
    currency: Optional[str] = None
    unit: Optional[str] = None  # millions, billions, etc.
    period: str  # "2023", "Q1 2023", etc.
    
    # Context
    context: Optional[str] = None  # Surrounding text
    confidence_score: Optional[float] = None
    
    class Config:
        use_enum_values = True


class QueryRequest(BaseModel):
    """Request model for financial queries."""
    
    question: str
    company: Optional[Company] = None
    year: Optional[int] = None
    section_type: Optional[SectionType] = None
    max_chunks: int = Field(10, ge=1, le=50)
    include_context: bool = True


class QueryResponse(BaseModel):
    """Response model for financial queries."""
    
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: Optional[float] = None
    disclaimer: Optional[str] = None
    processing_time: Optional[float] = None
    
    # Retrieved chunks for transparency
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)


class ComparisonRequest(BaseModel):
    """Request model for company comparisons."""
    
    companies: List[Company]
    metric: str
    years: List[int]
    normalize_currency: bool = True


class ComparisonResponse(BaseModel):
    """Response model for company comparisons."""
    
    comparison_data: Dict[str, Any]
    insights: str
    methodology: str
    disclaimers: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
