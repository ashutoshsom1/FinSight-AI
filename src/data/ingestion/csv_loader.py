"""CSV/Excel loader for financial data tables."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import structlog

from .base_loader import BaseDocumentLoader
from ..models.document import DocumentChunk, SectionType, DocumentMetadata

logger = structlog.get_logger()


class CSVLoader(BaseDocumentLoader):
    """Loader for CSV and Excel financial data."""
    
    def __init__(self, metadata: DocumentMetadata, chunk_size: int = 1000):
        super().__init__(metadata)
        self.chunk_size = chunk_size
    
    async def load(self) -> List[DocumentChunk]:
        """Load and parse CSV/Excel into chunks."""
        self.logger.info("Starting CSV/Excel loading", file_path=self.metadata.file_path)
        
        try:
            # Read the file
            df = self._read_file(Path(self.metadata.file_path))
            
            # Process the dataframe into chunks
            chunks = await self._process_dataframe(df)
            
            self.logger.info("CSV/Excel loading completed", total_chunks=len(chunks))
            return chunks
            
        except Exception as e:
            self.logger.error("Failed to load CSV/Excel", error=str(e))
            raise
    
    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read CSV or Excel file into DataFrame."""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            # For Excel files, we might have multiple sheets
            excel_file = pd.ExcelFile(file_path)
            
            # If multiple sheets, combine them or process separately
            if len(excel_file.sheet_names) == 1:
                return pd.read_excel(file_path)
            else:
                # Combine all sheets with sheet name as prefix
                combined_df = pd.DataFrame()
                for sheet_name in excel_file.sheet_names:
                    sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
                    sheet_df['sheet_name'] = sheet_name
                    combined_df = pd.concat([combined_df, sheet_df], ignore_index=True)
                return combined_df
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text representation from CSV/Excel."""
        df = self._read_file(file_path)
        return df.to_string()
    
    async def _process_dataframe(self, df: pd.DataFrame) -> List[DocumentChunk]:
        """Process DataFrame into meaningful chunks."""
        chunks = []
        
        # Strategy 1: Chunk by rows if large dataset
        if len(df) > 100:
            chunks.extend(await self._chunk_by_rows(df))
        
        # Strategy 2: Create summary chunk
        summary_chunk = await self._create_summary_chunk(df)
        if summary_chunk:
            chunks.append(summary_chunk)
        
        # Strategy 3: Create column-wise chunks for financial metrics
        metric_chunks = await self._create_metric_chunks(df)
        chunks.extend(metric_chunks)
        
        return chunks
    
    async def _chunk_by_rows(self, df: pd.DataFrame, rows_per_chunk: int = 50) -> List[DocumentChunk]:
        """Create chunks by grouping rows."""
        chunks = []
        
        for i in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[i:i + rows_per_chunk]
            
            # Convert to readable text format
            content = self._dataframe_to_text(chunk_df)
            
            chunk = DocumentChunk(
                document_id=self.metadata.document_id,
                content=content,
                section_type=SectionType.FINANCIAL_STATEMENTS,
                chunk_index=len(chunks),
                contains_tables=True,
                contains_numbers=True,
                financial_metrics=self._extract_metrics_from_columns(chunk_df.columns)
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _create_summary_chunk(self, df: pd.DataFrame) -> Optional[DocumentChunk]:
        """Create a summary chunk describing the dataset."""
        
        summary_parts = [
            f"Financial data table with {len(df)} rows and {len(df.columns)} columns.",
            f"Columns: {', '.join(df.columns.tolist())}",
        ]
        
        # Add data type information
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            summary_parts.append(f"Numeric columns: {', '.join(numeric_cols)}")
        
        # Add date range if date columns exist
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        if date_cols:
            for col in date_cols:
                min_date = df[col].min()
                max_date = df[col].max()
                summary_parts.append(f"Date range for {col}: {min_date} to {max_date}")
        
        # Add basic statistics for key financial metrics
        financial_keywords = ['revenue', 'income', 'profit', 'assets', 'equity', 'debt']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in financial_keywords):
                if df[col].dtype in ['int64', 'float64']:
                    stats = df[col].describe()
                    summary_parts.append(
                        f"{col}: Mean={stats['mean']:.2f}, "
                        f"Min={stats['min']:.2f}, Max={stats['max']:.2f}"
                    )
        
        content = '\n'.join(summary_parts)
        
        return DocumentChunk(
            document_id=self.metadata.document_id,
            content=content,
            section_type=SectionType.FINANCIAL_STATEMENTS,
            chunk_index=0,
            contains_tables=True,
            contains_numbers=True,
            financial_metrics=self._extract_metrics_from_columns(df.columns)
        )
    
    async def _create_metric_chunks(self, df: pd.DataFrame) -> List[DocumentChunk]:
        """Create focused chunks for specific financial metrics."""
        chunks = []
        
        # Group columns by financial metric type
        metric_groups = self._group_columns_by_metrics(df.columns)
        
        for metric_type, columns in metric_groups.items():
            if not columns:
                continue
            
            # Create a focused view of this metric
            metric_df = df[columns + self._get_identifier_columns(df)]
            content = self._dataframe_to_text(metric_df)
            
            chunk = DocumentChunk(
                document_id=self.metadata.document_id,
                content=f"Financial Metric: {metric_type.upper()}\n\n{content}",
                section_type=SectionType.FINANCIAL_STATEMENTS,
                chunk_index=len(chunks) + 1000,  # Offset to avoid conflicts
                contains_tables=True,
                contains_numbers=True,
                financial_metrics=[metric_type]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text format."""
        # Use a more readable format than default to_string()
        lines = []
        
        # Add header
        lines.append("| " + " | ".join(df.columns) + " |")
        lines.append("|" + "|".join(["-" * (len(col) + 2) for col in df.columns]) + "|")
        
        # Add rows
        for _, row in df.iterrows():
            formatted_row = []
            for value in row:
                if pd.isna(value):
                    formatted_row.append("-")
                elif isinstance(value, (int, float)):
                    formatted_row.append(f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
                else:
                    formatted_row.append(str(value))
            lines.append("| " + " | ".join(formatted_row) + " |")
        
        return '\n'.join(lines)
    
    def _extract_metrics_from_columns(self, columns: List[str]) -> List[str]:
        """Extract financial metrics from column names."""
        from config.settings import FINANCIAL_METRICS_MAPPING
        
        found_metrics = []
        for col in columns:
            col_lower = col.lower()
            for standard_name, variations in FINANCIAL_METRICS_MAPPING.items():
                if any(variation in col_lower for variation in variations):
                    if standard_name not in found_metrics:
                        found_metrics.append(standard_name)
        
        return found_metrics
    
    def _group_columns_by_metrics(self, columns: List[str]) -> Dict[str, List[str]]:
        """Group columns by financial metric types."""
        from config.settings import FINANCIAL_METRICS_MAPPING
        
        metric_groups = {metric: [] for metric in FINANCIAL_METRICS_MAPPING.keys()}
        
        for col in columns:
            col_lower = col.lower()
            for standard_name, variations in FINANCIAL_METRICS_MAPPING.items():
                if any(variation in col_lower for variation in variations):
                    metric_groups[standard_name].append(col)
                    break
        
        # Remove empty groups
        return {k: v for k, v in metric_groups.items() if v}
    
    def _get_identifier_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that serve as identifiers (year, quarter, etc.)."""
        identifier_keywords = ['year', 'quarter', 'period', 'date', 'time']
        identifier_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in identifier_keywords):
                identifier_cols.append(col)
        
        return identifier_cols
