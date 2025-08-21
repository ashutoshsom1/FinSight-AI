"""Factory for creating embedding services based on configuration."""

from typing import Union
import structlog
from config.settings import settings

logger = structlog.get_logger()


def create_embedding_service() -> Union["EmbeddingService", "AzureEmbeddingService"]:
    """Create embedding service based on configuration."""
    
    llm_config = settings.get_llm_config()
    
    if llm_config["provider"] == "openai":
        from .embedding_service import EmbeddingService
        
        return EmbeddingService(
            api_key=llm_config["api_key"],
            model=llm_config["embedding_model"]
        )
    
    elif llm_config["provider"] == "azure":
        from .azure_embedding_service import AzureEmbeddingService
        
        return AzureEmbeddingService(
            endpoint=llm_config["endpoint"],
            api_key=llm_config["api_key"],
            api_version=llm_config.get("api_version", "2024-10-21"),
            deployment_name=llm_config["embedding_deployment"]
        )
    
    else:
        raise ValueError(f"Unsupported embedding provider: {llm_config['provider']}")


async def test_embedding_service() -> bool:
    """Test the embedding service connection."""
    
    try:
        service = create_embedding_service()
        
        # Test with a simple embedding
        test_embedding = await service.generate_single_embedding("test connection")
        
        if len(test_embedding) > 0:
            logger.info("Embedding service test successful", 
                       provider=settings.llm_provider,
                       dimension=len(test_embedding))
            return True
        else:
            logger.error("Embedding service test failed - empty embedding returned")
            return False
            
    except Exception as e:
        logger.error("Embedding service test failed", error=str(e))
        return False
