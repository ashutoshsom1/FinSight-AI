"""Script to ingest financial reports into the knowledge base."""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import structlog

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.models.document import DocumentMetadata, DocumentType, Company
from data.ingestion.pdf_loader import PDFLoader
from data.ingestion.csv_loader import CSVLoader
from data.storage.vector_store import create_vector_store
from data.storage.blob_store import create_blob_store
from embeddings.embedding_service import EmbeddingService
from preprocessing.text_cleaner import FinancialTextCleaner
from config.settings import settings

logger = structlog.get_logger()


class DocumentIngestionPipeline:
    """Pipeline for ingesting financial documents."""
    
    def __init__(self):
        self.text_cleaner = FinancialTextCleaner()
        self.vector_store = None
        self.blob_store = None
        self.embedding_service = None
    
    async def initialize(self):
        """Initialize all services."""
        logger.info("Initializing ingestion pipeline")
        
        self.vector_store = await create_vector_store(
            settings.vector_db_type,
            persist_directory=settings.chroma_persist_directory
        )
        
        self.blob_store = await create_blob_store(
            settings.blob_storage_type,
            base_path="./data/documents"
        )
        
        self.embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model
        )
        
        logger.info("Pipeline initialization completed")
    
    async def ingest_document(
        self, 
        file_path: str, 
        company: str, 
        year: int,
        document_type: str = "annual_report"
    ) -> Dict[str, Any]:
        """Ingest a single document."""
        
        logger.info("Starting document ingestion", file_path=file_path, company=company, year=year)
        
        try:
            # Create metadata
            metadata = DocumentMetadata(
                company=Company(company),
                document_type=DocumentType(document_type),
                year=year,
                file_path=file_path
            )
            
            # Choose appropriate loader
            file_path_obj = Path(file_path)
            if file_path_obj.suffix.lower() == '.pdf':
                loader = PDFLoader(metadata)
            elif file_path_obj.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                loader = CSVLoader(metadata)
            else:
                raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")
            
            # Load document
            chunks = await loader.load()
            logger.info("Document loaded", chunk_count=len(chunks))
            
            # Clean chunks
            for chunk in chunks:
                chunk.content = self.text_cleaner.clean_text(chunk.content)
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.generate_embeddings(chunk_texts)
            logger.info("Embeddings generated", embedding_count=len(embeddings))
            
            # Store in vector database
            await self.vector_store.add_chunks(chunks, embeddings)
            logger.info("Chunks stored in vector database")
            
            # Store original file in blob storage
            blob_name = f"{company}/{year}/{file_path_obj.name}"
            blob_url = await self.blob_store.upload_file(
                file_path=file_path_obj,
                blob_name=blob_name,
                metadata={
                    "company": company,
                    "year": year,
                    "document_type": document_type
                }
            )
            logger.info("Original file stored in blob storage", blob_url=blob_url)
            
            return {
                "status": "success",
                "document_id": metadata.document_id,
                "chunks_created": len(chunks),
                "blob_url": blob_url
            }
            
        except Exception as e:
            logger.error("Document ingestion failed", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def ingest_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Ingest all supported documents from a directory."""
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        results = []
        supported_extensions = ['.pdf', '.csv', '.xlsx', '.xls']
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                # Try to extract company and year from filename or path
                company, year = self._extract_metadata_from_path(file_path)
                
                if company and year:
                    result = await self.ingest_document(
                        str(file_path), company, year
                    )
                    results.append({
                        "file": str(file_path),
                        "company": company,
                        "year": year,
                        "result": result
                    })
                else:
                    logger.warning("Could not extract metadata from file path", file_path=str(file_path))
        
        return results
    
    def _extract_metadata_from_path(self, file_path: Path) -> tuple[Optional[str], Optional[int]]:
        """Extract company and year from file path."""
        
        path_str = str(file_path).lower()
        
        # Extract company
        company = None
        for target_company in settings.get_target_companies_list():
            if target_company in path_str:
                company = target_company
                break
        
        # Extract year
        year = None
        import re
        year_matches = re.findall(r'\b(20\d{2})\b', path_str)
        if year_matches:
            year = int(year_matches[-1])  # Take the last year found
        
        return company, year


async def main():
    """Main function for running document ingestion."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest financial documents")
    parser.add_argument("--file", help="Single file to ingest")
    parser.add_argument("--directory", help="Directory to ingest")
    parser.add_argument("--company", help="Company name (castrol, veedol, valvoline)")
    parser.add_argument("--year", type=int, help="Document year")
    parser.add_argument("--type", default="annual_report", help="Document type")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentIngestionPipeline()
    await pipeline.initialize()
    
    if args.file:
        if not args.company or not args.year:
            print("Error: --company and --year are required when ingesting a single file")
            return
        
        result = await pipeline.ingest_document(
            args.file, args.company, args.year, args.type
        )
        print(f"Ingestion result: {result}")
    
    elif args.directory:
        results = await pipeline.ingest_directory(args.directory)
        print(f"Ingested {len(results)} documents")
        for result in results:
            print(f"  {result['file']}: {result['result']['status']}")
    
    else:
        print("Error: Either --file or --directory must be specified")


if __name__ == "__main__":
    asyncio.run(main())
