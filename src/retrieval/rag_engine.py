"""Retrieval-Augmented Generation engine for financial queries."""

from typing import List, Dict, Any, Optional, Tuple
import structlog
from datetime import datetime

from ..data.models.document import DocumentChunk, QueryRequest, QueryResponse, Company
from ..data.storage.vector_store import VectorStore
from ..embeddings.embedding_service import EmbeddingService

logger = structlog.get_logger()


class RAGEngine:
    """Core RAG engine for financial document retrieval and response generation."""
    
    def __init__(
        self, 
        vector_store: VectorStore, 
        embedding_service: EmbeddingService,
        max_context_tokens: int = 8000
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.max_context_tokens = max_context_tokens
        self.logger = logger.bind(component="rag_engine")
    
    async def retrieve_relevant_chunks(
        self, 
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 10
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant document chunks for a query."""
        
        self.logger.info("Starting chunk retrieval", query=query[:100])
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(query)
            
            # Search vector store
            chunks_with_scores = await self.vector_store.search(
                query_embedding=query_embedding,
                k=k,
                filters=filters
            )
            
            # Filter and rank results
            filtered_chunks = await self._filter_and_rank_chunks(
                chunks_with_scores, query
            )
            
            self.logger.info(
                "Chunk retrieval completed", 
                retrieved_count=len(filtered_chunks)
            )
            
            return filtered_chunks
            
        except Exception as e:
            self.logger.error("Failed to retrieve chunks", error=str(e))
            raise
    
    async def _filter_and_rank_chunks(
        self, 
        chunks_with_scores: List[Tuple[DocumentChunk, float]],
        query: str
    ) -> List[Tuple[DocumentChunk, float]]:
        """Apply additional filtering and ranking to retrieved chunks."""
        
        # Filter by minimum similarity threshold
        min_similarity = 0.7
        filtered_chunks = [
            (chunk, score) for chunk, score in chunks_with_scores 
            if score >= min_similarity
        ]
        
        # Re-rank based on query-specific criteria
        reranked_chunks = await self._rerank_chunks(filtered_chunks, query)
        
        return reranked_chunks
    
    async def _rerank_chunks(
        self, 
        chunks_with_scores: List[Tuple[DocumentChunk, float]],
        query: str
    ) -> List[Tuple[DocumentChunk, float]]:
        """Re-rank chunks based on query-specific criteria."""
        
        query_lower = query.lower()
        
        # Boost scores based on query characteristics
        boosted_chunks = []
        
        for chunk, score in chunks_with_scores:
            boost_factor = 1.0
            
            # Boost chunks with financial metrics mentioned in query
            for metric in chunk.financial_metrics:
                if metric.replace('_', ' ') in query_lower:
                    boost_factor += 0.2
            
            # Boost chunks with tables if query asks for numbers/data
            if chunk.contains_tables and any(word in query_lower for word in ['data', 'numbers', 'table', 'figure']):
                boost_factor += 0.15
            
            # Boost chunks with numbers if query asks for specific values
            if chunk.contains_numbers and any(word in query_lower for word in ['how much', 'what is', 'value', 'amount']):
                boost_factor += 0.1
            
            # Boost recent sections for trend queries
            if any(word in query_lower for word in ['trend', 'growth', 'change', 'increase', 'decrease']):
                if chunk.section_type.value in ['income_statement', 'financial_statements']:
                    boost_factor += 0.1
            
            boosted_score = min(score * boost_factor, 1.0)  # Cap at 1.0
            boosted_chunks.append((chunk, boosted_score))
        
        # Sort by boosted scores
        boosted_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return boosted_chunks
    
    async def build_context(
        self, 
        chunks_with_scores: List[Tuple[DocumentChunk, float]],
        query: str
    ) -> Dict[str, Any]:
        """Build context for LLM from retrieved chunks."""
        
        context_parts = []
        total_tokens = 0
        used_chunks = []
        sources = []
        
        for chunk, score in chunks_with_scores:
            # Estimate tokens for this chunk
            chunk_tokens = chunk.token_count or len(chunk.content.split()) * 1.3
            
            # Check if adding this chunk would exceed token limit
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
            
            # Format chunk for context
            chunk_context = self._format_chunk_for_context(chunk, score)
            context_parts.append(chunk_context)
            
            total_tokens += chunk_tokens
            used_chunks.append(chunk)
            
            # Add source information
            sources.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "section_type": chunk.section_type,
                "page_number": chunk.page_number,
                "similarity_score": score,
                "financial_metrics": chunk.financial_metrics
            })
        
        context = {
            "formatted_context": "\n\n---\n\n".join(context_parts),
            "chunks": used_chunks,
            "sources": sources,
            "total_tokens": total_tokens,
            "chunk_count": len(used_chunks)
        }
        
        self.logger.debug(
            "Context built", 
            chunk_count=len(used_chunks),
            total_tokens=total_tokens
        )
        
        return context
    
    def _format_chunk_for_context(self, chunk: DocumentChunk, score: float) -> str:
        """Format a chunk for inclusion in LLM context."""
        
        # Create header with metadata
        header_parts = [f"Section: {chunk.section_type.value.replace('_', ' ').title()}"]
        
        if chunk.page_number:
            header_parts.append(f"Page: {chunk.page_number}")
        
        if chunk.financial_metrics:
            metrics_str = ", ".join(chunk.financial_metrics)
            header_parts.append(f"Metrics: {metrics_str}")
        
        header = f"[{' | '.join(header_parts)}]"
        
        # Format content
        content = chunk.content
        
        # Add table indicator if applicable
        if chunk.contains_tables:
            content = "[TABLE DATA]\n" + content
        
        return f"{header}\n{content}"
    
    async def generate_query_filters(self, request: QueryRequest) -> Dict[str, Any]:
        """Generate filters for vector search based on query request."""
        
        filters = {}
        
        # Company filter
        if request.company:
            # We'll need to map this to document_id patterns or add company to chunk metadata
            # For now, we'll handle this in the retrieval logic
            pass
        
        # Year filter (would need to be added to chunk metadata)
        if request.year:
            # This would require storing year information in chunk metadata
            pass
        
        # Section type filter
        if request.section_type:
            filters["section_type"] = request.section_type.value
        
        return filters
    
    async def validate_retrieval_quality(
        self, 
        query: str,
        chunks_with_scores: List[Tuple[DocumentChunk, float]]
    ) -> Dict[str, Any]:
        """Validate the quality of retrieved chunks."""
        
        if not chunks_with_scores:
            return {
                "quality_score": 0.0,
                "issues": ["No chunks retrieved"],
                "recommendations": ["Try broader search terms", "Check if documents are properly indexed"]
            }
        
        # Calculate quality metrics
        avg_similarity = sum(score for _, score in chunks_with_scores) / len(chunks_with_scores)
        max_similarity = max(score for _, score in chunks_with_scores)
        min_similarity = min(score for _, score in chunks_with_scores)
        
        # Check for diversity in sources
        unique_documents = len(set(chunk.document_id for chunk, _ in chunks_with_scores))
        unique_sections = len(set(chunk.section_type for chunk, _ in chunks_with_scores))
        
        # Identify potential issues
        issues = []
        recommendations = []
        
        if avg_similarity < 0.75:
            issues.append("Low average similarity scores")
            recommendations.append("Consider expanding search terms or checking document quality")
        
        if unique_documents == 1:
            issues.append("All results from single document")
            recommendations.append("Ensure multiple relevant documents are indexed")
        
        if max_similarity < 0.8:
            issues.append("No highly relevant chunks found")
            recommendations.append("Verify query matches document content")
        
        quality_score = (avg_similarity + (unique_documents / 10) + (unique_sections / 10)) / 3
        
        return {
            "quality_score": min(quality_score, 1.0),
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "unique_documents": unique_documents,
            "unique_sections": unique_sections,
            "issues": issues,
            "recommendations": recommendations
        }
