"""Embedding service for generating and managing document embeddings."""

import asyncio
from typing import List, Dict, Any, Optional
import structlog
import openai
from openai import AsyncOpenAI
import numpy as np
from datetime import datetime

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating embeddings using OpenAI's embedding models."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.logger = logger.bind(component="embedding_service", model=model)
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        
        self.logger.info("Starting embedding generation", text_count=len(texts))
        
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._generate_batch_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to be respectful of rate limits
            if i + self.batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        self.logger.info("Embedding generation completed", total_embeddings=len(all_embeddings))
        return all_embeddings
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                # Clean texts before embedding
                cleaned_texts = [self._prepare_text_for_embedding(text) for text in texts]
                
                # Call OpenAI API
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=cleaned_texts
                )
                
                # Extract embeddings
                embeddings = [data.embedding for data in response.data]
                
                self.logger.debug(
                    "Batch embedding successful", 
                    batch_size=len(texts),
                    attempt=attempt + 1
                )
                
                return embeddings
                
            except openai.RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.warning(
                    "Rate limit hit, retrying", 
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)
                
            except openai.APIError as e:
                self.logger.error(
                    "OpenAI API error", 
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(
                    "Unexpected error in embedding generation",
                    attempt=attempt + 1,
                    error=str(e)
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1)
        
        raise Exception(f"Failed to generate embeddings after {self.max_retries} attempts")
    
    def _prepare_text_for_embedding(self, text: str) -> str:
        """Prepare text for embedding by cleaning and truncating if necessary."""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Truncate if too long (OpenAI has token limits)
        # text-embedding-3-large has 8192 token limit
        max_tokens = 8000  # Leave some buffer
        
        if self._estimate_token_count(cleaned) > max_tokens:
            # Truncate to approximate token limit
            words = cleaned.split()
            # Rough estimate: 1 token ≈ 0.75 words
            max_words = int(max_tokens * 0.75)
            cleaned = ' '.join(words[:max_words])
            
            self.logger.warning(
                "Text truncated for embedding", 
                original_length=len(text),
                truncated_length=len(cleaned)
            )
        
        return cleaned
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters for English
        return len(text) // 4
    
    async def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    async def find_similar_chunks(
        self, 
        query_embedding: List[float],
        chunk_embeddings: List[Tuple[str, List[float]]],  # (chunk_id, embedding)
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find most similar chunks to a query embedding."""
        
        similarities = []
        
        for chunk_id, chunk_embedding in chunk_embeddings:
            similarity = await self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        
        return model_dimensions.get(self.model, 1536)  # Default to ada-002 dimension
    
    async def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding has the correct format and dimension."""
        
        if not isinstance(embedding, list):
            return False
        
        if len(embedding) != self.get_embedding_dimension():
            return False
        
        # Check that all values are valid floats
        try:
            for value in embedding:
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    return False
        except:
            return False
        
        return True
