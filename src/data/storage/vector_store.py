"""Vector storage implementation with multiple backend support."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import structlog
import asyncio
from pathlib import Path

from ..models.document import DocumentChunk

logger = structlog.get_logger()


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add document chunks with their embeddings to the store."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks and return with similarity scores."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a specific document."""
        pass
    
    @abstractmethod
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by ID."""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector storage."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "financial_documents"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.logger = logger.bind(store_type="chroma")
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Financial document chunks with embeddings"}
            )
            
            self.logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB", error=str(e))
            raise
    
    async def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add chunks with embeddings to ChromaDB."""
        if not self.collection:
            await self.initialize()
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [self._chunk_to_metadata(chunk) for chunk in chunks]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.logger.info("Added chunks to ChromaDB", count=len(chunks))
            
        except Exception as e:
            self.logger.error("Failed to add chunks to ChromaDB", error=str(e))
            raise
    
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search ChromaDB for similar chunks."""
        if not self.collection:
            await self.initialize()
        
        try:
            # Build where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        where_clause[key] = value
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause if where_clause else None
            )
            
            # Convert results to chunks with scores
            chunks_with_scores = []
            
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    content = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score (1 - distance for cosine)
                    similarity_score = 1.0 - distance
                    
                    # Reconstruct chunk from metadata
                    chunk = self._metadata_to_chunk(chunk_id, content, metadata)
                    chunks_with_scores.append((chunk, similarity_score))
            
            self.logger.info("Search completed", results_count=len(chunks_with_scores))
            return chunks_with_scores
            
        except Exception as e:
            self.logger.error("Failed to search ChromaDB", error=str(e))
            raise
    
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document from ChromaDB."""
        if not self.collection:
            await self.initialize()
        
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                self.logger.info("Deleted document chunks", document_id=document_id, count=len(results['ids']))
            
        except Exception as e:
            self.logger.error("Failed to delete document", document_id=document_id, error=str(e))
            raise
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        if not self.collection:
            await self.initialize()
        
        try:
            results = self.collection.get(ids=[chunk_id])
            
            if results['ids'] and results['ids'][0]:
                metadata = results['metadatas'][0]
                content = results['documents'][0]
                return self._metadata_to_chunk(chunk_id, content, metadata)
            
            return None
            
        except Exception as e:
            self.logger.error("Failed to get chunk", chunk_id=chunk_id, error=str(e))
            return None
    
    def _chunk_to_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert DocumentChunk to ChromaDB metadata format."""
        return {
            "document_id": chunk.document_id,
            "section_type": chunk.section_type,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "contains_tables": chunk.contains_tables,
            "contains_numbers": chunk.contains_numbers,
            "financial_metrics": ",".join(chunk.financial_metrics) if chunk.financial_metrics else "",
        }
    
    def _metadata_to_chunk(self, chunk_id: str, content: str, metadata: Dict[str, Any]) -> DocumentChunk:
        """Convert ChromaDB metadata back to DocumentChunk."""
        financial_metrics = []
        if metadata.get("financial_metrics"):
            financial_metrics = metadata["financial_metrics"].split(",")
        
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=metadata["document_id"],
            content=content,
            section_type=metadata["section_type"],
            page_number=metadata.get("page_number"),
            chunk_index=metadata["chunk_index"],
            token_count=metadata.get("token_count"),
            contains_tables=metadata.get("contains_tables", False),
            contains_numbers=metadata.get("contains_numbers", False),
            financial_metrics=financial_metrics,
        )


# Factory function for vector store creation
async def create_vector_store(store_type: str, **kwargs) -> VectorStore:
    """Factory function to create vector store instances."""

    if store_type.lower() == "chroma":
        store = ChromaVectorStore(**kwargs)
        await store.initialize()
        return store
    elif store_type.lower() == "pinecone":
        # Placeholder for Pinecone implementation
        raise NotImplementedError("Pinecone implementation coming soon")
    elif store_type.lower() == "weaviate":
        # Placeholder for Weaviate implementation
        raise NotImplementedError("Weaviate implementation coming soon")
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")
