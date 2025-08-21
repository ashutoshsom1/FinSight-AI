"""Factory for creating LLM services based on configuration."""

from typing import Union
import structlog
from config.settings import settings

logger = structlog.get_logger()


def create_financial_llm() -> Union["FinancialLLM", "AzureFinancialLLM"]:
    """Create financial LLM service based on configuration."""
    
    llm_config = settings.get_llm_config()
    
    if llm_config["provider"] == "openai":
        from .financial_llm import FinancialLLM
        
        return FinancialLLM(
            api_key=llm_config["api_key"],
            model=llm_config["model"]
        )
    
    elif llm_config["provider"] == "azure":
        from .azure_financial_llm import AzureFinancialLLM
        
        return AzureFinancialLLM(
            endpoint=llm_config["endpoint"],
            api_key=llm_config["api_key"],
            api_version=llm_config.get("api_version", "2024-10-21"),
            deployment_name=llm_config["deployment_name"]
        )
    
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config['provider']}")


async def test_financial_llm() -> bool:
    """Test the financial LLM service connection."""
    
    try:
        service = create_financial_llm()
        
        # Test connection if method exists
        if hasattr(service, 'test_connection'):
            return await service.test_connection()
        else:
            # For standard OpenAI service, we can't easily test without making a real call
            logger.info("LLM service created successfully", provider=settings.llm_provider)
            return True
            
    except Exception as e:
        logger.error("LLM service test failed", error=str(e))
        return False
