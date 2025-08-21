"""Core financial service orchestrating all components."""

from typing import List, Dict, Any, Optional
import structlog
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.models.document import (
    QueryRequest, QueryResponse, ComparisonRequest, ComparisonResponse,
    DocumentMetadata, DocumentType, Company
)
from data.storage.vector_store import VectorStore
from data.storage.blob_store import BlobStore
from data.ingestion.pdf_loader import PDFLoader
from data.ingestion.csv_loader import CSVLoader
from preprocessing.text_cleaner import FinancialTextCleaner
from preprocessing.chunker import SemanticChunker, ChunkingConfig
from retrieval.rag_engine import RAGEngine
from llm.financial_llm import FinancialLLM
from config.settings import settings, COMPANY_CONFIGS

logger = structlog.get_logger()


class FinancialService:
    """Main service orchestrating financial analysis workflows."""
    
    def __init__(
        self, 
        rag_engine: RAGEngine,
        financial_llm: FinancialLLM,
        blob_store: BlobStore
    ):
        self.rag_engine = rag_engine
        self.financial_llm = financial_llm
        self.blob_store = blob_store
        self.text_cleaner = FinancialTextCleaner()
        self.chunker = SemanticChunker(
            ChunkingConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
        )
        self.logger = logger.bind(component="financial_service")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a financial query using RAG."""
        
        self.logger.info("Processing financial query", question=request.question[:100])
        
        try:
            # Generate filters based on request
            filters = await self._build_query_filters(request)
            
            # Retrieve relevant chunks
            chunks_with_scores = await self.rag_engine.retrieve_relevant_chunks(
                query=request.question,
                filters=filters,
                k=request.max_chunks
            )
            
            if not chunks_with_scores:
                return QueryResponse(
                    answer="I couldn't find relevant information in the available financial documents to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    disclaimer=self.financial_llm._get_disclaimer() if settings.include_disclaimers else None
                )
            
            # Build context for LLM
            context = await self.rag_engine.build_context(chunks_with_scores, request.question)
            
            # Generate response using LLM
            response = await self.financial_llm.generate_financial_response(
                query=request.question,
                context=context,
                include_disclaimer=settings.include_disclaimers
            )
            
            self.logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            self.logger.error("Failed to process query", error=str(e))
            raise
    
    async def compare_companies(self, request: ComparisonRequest) -> ComparisonResponse:
        """Compare financial metrics across companies."""
        
        self.logger.info("Processing company comparison", companies=request.companies, metric=request.metric)
        
        try:
            # Retrieve data for each company
            comparison_data = {}
            all_sources = []
            
            for company in request.companies:
                company_data = await self._get_company_metric_data(
                    company=company,
                    metric=request.metric,
                    years=request.years
                )
                comparison_data[company.value] = company_data["data"]
                all_sources.extend(company_data["sources"])
            
            # Generate comparison insights
            response = await self.financial_llm.generate_comparison_response(
                companies=[c.value for c in request.companies],
                metric=request.metric,
                context_data={
                    "comparison_data": comparison_data,
                    "sources": all_sources,
                    "normalize_currency": request.normalize_currency
                }
            )
            
            self.logger.info("Company comparison completed")
            return response
            
        except Exception as e:
            self.logger.error("Failed to process comparison", error=str(e))
            raise
    
    async def generate_summary(
        self, 
        company: str, 
        year: int, 
        section_type: Optional[str] = None
    ) -> str:
        """Generate a summary of company's financial performance."""
        
        self.logger.info("Generating summary", company=company, year=year)
        
        try:
            # Build query for summary data
            if section_type:
                query = f"Summarize {company}'s {section_type} for {year}"
            else:
                query = f"Provide an overview of {company}'s financial performance in {year}"
            
            # Retrieve relevant chunks
            filters = {"company": company} if company in [c.value for c in Company] else {}
            
            chunks_with_scores = await self.rag_engine.retrieve_relevant_chunks(
                query=query,
                filters=filters,
                k=15  # More chunks for comprehensive summary
            )
            
            if not chunks_with_scores:
                return f"No financial data available for {company} in {year}"
            
            # Generate summary using LLM
            context = await self.rag_engine.build_context(chunks_with_scores, query)
            response = await self.financial_llm.generate_financial_response(
                query=query,
                context=context,
                include_disclaimer=settings.include_disclaimers
            )
            
            return response.answer
            
        except Exception as e:
            self.logger.error("Failed to generate summary", error=str(e))
            raise
    
    async def ingest_document(
        self, 
        file_path: str, 
        company: str, 
        year: int,
        document_type: str = "annual_report"
    ) -> Dict[str, Any]:
        """Ingest a new financial document into the knowledge base."""
        
        self.logger.info("Starting document ingestion", file_path=file_path, company=company, year=year)
        
        try:
            # Create document metadata
            metadata = DocumentMetadata(
                company=Company(company),
                document_type=DocumentType(document_type),
                year=year,
                file_path=file_path,
                processed_at=datetime.now()
            )
            
            # Determine loader based on file extension
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.pdf':
                loader = PDFLoader(metadata, settings.chunk_size, settings.chunk_overlap)
            elif file_path_obj.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                loader = CSVLoader(metadata, settings.chunk_size)
            else:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
            
            # Load and process document
            chunks = await loader.load()
            
            # Clean and re-chunk if necessary
            processed_chunks = []
            for chunk in chunks:
                cleaned_content = self.text_cleaner.clean_text(chunk.content)
                chunk.content = cleaned_content
                processed_chunks.append(chunk)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in processed_chunks]
            embeddings = await self.rag_engine.embedding_service.generate_embeddings(chunk_texts)
            
            # Store in vector database
            await self.rag_engine.vector_store.add_chunks(processed_chunks, embeddings)
            
            # Store original document in blob storage
            blob_name = f"{company}/{year}/{file_path_obj.name}"
            blob_url = await self.blob_store.upload_file(
                file_path=file_path_obj,
                blob_name=blob_name,
                metadata={
                    "company": company,
                    "year": year,
                    "document_type": document_type,
                    "processed_at": datetime.now().isoformat()
                }
            )
            
            result = {
                "document_id": metadata.document_id,
                "chunks_created": len(processed_chunks),
                "blob_url": blob_url,
                "status": "completed"
            }
            
            self.logger.info("Document ingestion completed", **result)
            return result
            
        except Exception as e:
            self.logger.error("Failed to ingest document", error=str(e))
            raise
    
    async def list_documents(
        self, 
        company: Optional[str] = None, 
        year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List available documents in the knowledge base."""
        
        try:
            # Get files from blob storage
            prefix = None
            if company:
                prefix = company
                if year:
                    prefix = f"{company}/{year}"
            
            files = await self.blob_store.list_files(prefix=prefix)
            
            return files
            
        except Exception as e:
            self.logger.error("Failed to list documents", error=str(e))
            raise
    
    async def _build_query_filters(self, request: QueryRequest) -> Dict[str, Any]:
        """Build filters for vector search based on query request."""
        
        filters = {}
        
        # Note: These filters assume we store company/year info in chunk metadata
        # This would need to be implemented in the ingestion pipeline
        
        if request.company:
            filters["company"] = request.company.value
        
        if request.year:
            filters["year"] = request.year
        
        if request.section_type:
            filters["section_type"] = request.section_type.value
        
        return filters
    
    async def _get_company_metric_data(
        self, 
        company: Company, 
        metric: str, 
        years: List[int]
    ) -> Dict[str, Any]:
        """Retrieve specific metric data for a company across years."""
        
        # Build query for the specific metric
        query = f"{company.value} {metric} " + " ".join(map(str, years))
        
        # Retrieve relevant chunks
        filters = {"company": company.value}
        chunks_with_scores = await self.rag_engine.retrieve_relevant_chunks(
            query=query,
            filters=filters,
            k=20  # More chunks for comprehensive data
        )
        
        # Extract metric data from chunks
        metric_data = {}
        sources = []
        
        for chunk, score in chunks_with_scores:
            # Use LLM to extract specific metric values
            extracted_metrics = await self.financial_llm.extract_key_metrics(
                chunk.content, 
                company.value
            )
            
            # Store data by year if available
            for year in years:
                if str(year) in chunk.content and metric in extracted_metrics:
                    if year not in metric_data:
                        metric_data[year] = []
                    metric_data[year].append({
                        "value": extracted_metrics[metric],
                        "source_chunk": chunk.chunk_id,
                        "confidence": score
                    })
            
            sources.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "similarity_score": score
            })
        
        return {
            "data": metric_data,
            "sources": sources
        }
