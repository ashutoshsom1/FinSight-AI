"""FastAPI main application for FinSight AI."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from contextlib import asynccontextmanager
from typing import Dict, Any

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.models.document import QueryRequest, QueryResponse, ComparisonRequest, ComparisonResponse
from data.storage.vector_store import create_vector_store
from data.storage.blob_store import create_blob_store
from embeddings.embedding_service import EmbeddingService
from retrieval.rag_engine import RAGEngine
from llm.financial_llm import FinancialLLM
from api.services.financial_service import FinancialService
from api.middleware.logging import setup_logging
from config.settings import settings

# Setup logging
setup_logging()
logger = structlog.get_logger()

# Global service instances
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    
    logger.info("Starting FinSight AI application")
    
    try:
        # Initialize services
        vector_store = await create_vector_store(
            settings.vector_db_type,
            persist_directory=settings.chroma_persist_directory,
            collection_name="financial_documents"
        )
        
        blob_store = await create_blob_store(
            settings.blob_storage_type,
            base_path="./data/documents"
        )
        
        embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
        
        rag_engine = RAGEngine(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        
        financial_llm = FinancialLLM(
            api_key=settings.openai_api_key,
            model=settings.openai_model
        )
        
        financial_service = FinancialService(
            rag_engine=rag_engine,
            financial_llm=financial_llm,
            blob_store=blob_store
        )
        
        # Store in app state
        app_state.update({
            "vector_store": vector_store,
            "blob_store": blob_store,
            "embedding_service": embedding_service,
            "rag_engine": rag_engine,
            "financial_llm": financial_llm,
            "financial_service": financial_service
        })
        
        logger.info("Application startup completed")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="FinSight AI",
    description="LLM-powered knowledge base for financial insights",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_financial_service() -> FinancialService:
    """Dependency to get financial service instance."""
    return app_state["financial_service"]


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FinSight AI",
        "version": "0.1.0",
        "description": "LLM-powered knowledge base for financial insights",
        "companies": settings.get_target_companies_list(),
        "supported_years": settings.get_supported_years_list(),
        "endpoints": {
            "query": "/query - Ask financial questions",
            "compare": "/compare - Compare companies",
            "summary": "/summary - Generate summaries",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    try:
        # Check if services are initialized
        required_services = ["financial_service", "vector_store", "embedding_service"]
        missing_services = [svc for svc in required_services if svc not in app_state]
        
        if missing_services:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "missing_services": missing_services
                }
            )
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": list(app_state.keys())
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/query", response_model=QueryResponse)
async def query_financial_data(
    request: QueryRequest,
    financial_service: FinancialService = Depends(get_financial_service)
):
    """Query financial data and get AI-powered insights."""
    
    logger.info("Received financial query", question=request.question[:100])
    
    try:
        response = await financial_service.process_query(request)
        return response
        
    except Exception as e:
        logger.error("Failed to process query", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/compare", response_model=ComparisonResponse)
async def compare_companies(
    request: ComparisonRequest,
    financial_service: FinancialService = Depends(get_financial_service)
):
    """Compare financial metrics across companies."""
    
    logger.info("Received comparison request", companies=request.companies, metric=request.metric)
    
    try:
        response = await financial_service.compare_companies(request)
        return response
        
    except Exception as e:
        logger.error("Failed to process comparison", error=str(e))
        raise HTTPException(status_code=500, detail=f"Comparison processing failed: {str(e)}")


@app.post("/summary")
async def generate_summary(
    company: str,
    year: int,
    section_type: Optional[str] = None,
    financial_service: FinancialService = Depends(get_financial_service)
):
    """Generate summary of company's financial performance."""
    
    logger.info("Received summary request", company=company, year=year)
    
    try:
        summary = await financial_service.generate_summary(
            company=company,
            year=year,
            section_type=section_type
        )
        return {"summary": summary}
        
    except Exception as e:
        logger.error("Failed to generate summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file_path: str,
    company: str,
    year: int,
    document_type: str = "annual_report",
    financial_service: FinancialService = Depends(get_financial_service)
):
    """Ingest a new financial document."""
    
    logger.info("Received ingestion request", file_path=file_path, company=company, year=year)
    
    try:
        # Add ingestion task to background
        background_tasks.add_task(
            financial_service.ingest_document,
            file_path=file_path,
            company=company,
            year=year,
            document_type=document_type
        )
        
        return {
            "message": "Document ingestion started",
            "file_path": file_path,
            "company": company,
            "year": year,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error("Failed to start document ingestion", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/documents")
async def list_documents(
    company: Optional[str] = None,
    year: Optional[int] = None,
    financial_service: FinancialService = Depends(get_financial_service)
):
    """List available documents in the knowledge base."""
    
    try:
        documents = await financial_service.list_documents(
            company=company,
            year=year
        )
        return {"documents": documents}
        
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Document listing failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
