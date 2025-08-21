"""Configuration settings for FinSight AI."""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Provider Configuration
    llm_provider: str = Field("openai", env="LLM_PROVIDER")  # "openai" or "azure"

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")

    # Azure OpenAI Configuration
    azure_openai_endpoint: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: Optional[str] = Field(None, env="AZURE_OPENAI_KEY")
    azure_openai_api_version: str = Field("2024-10-21", env="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment_name: str = Field("gpt-35-turbo-16k", env="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_embedding_deployment: str = Field("text-embedding-ada-002", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    
    # Vector Database Configuration
    vector_db_type: str = Field("chroma", env="VECTOR_DB_TYPE")
    chroma_persist_directory: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("finsight-financial-kb", env="PINECONE_INDEX_NAME")
    
    # Storage Configuration
    blob_storage_type: str = Field("local", env="BLOB_STORAGE_TYPE")
    azure_storage_connection_string: Optional[str] = Field(None, env="AZURE_STORAGE_CONNECTION_STRING")
    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    aws_bucket_name: str = Field("finsight-documents", env="AWS_BUCKET_NAME")
    
    # Application Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Chunking Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    max_chunks_per_query: int = Field(10, env="MAX_CHUNKS_PER_QUERY")
    
    # Financial Data Configuration
    target_companies: List[str] = Field(
        ["castrol", "veedol", "valvoline"], 
        env="TARGET_COMPANIES"
    )
    supported_years: List[int] = Field(
        [2020, 2021, 2022, 2023, 2024], 
        env="SUPPORTED_YEARS"
    )
    
    # Evaluation Configuration
    enable_ragas_evaluation: bool = Field(True, env="ENABLE_RAGAS_EVALUATION")
    feedback_storage_path: str = Field("./data/feedback", env="FEEDBACK_STORAGE_PATH")
    
    # Compliance Configuration
    include_disclaimers: bool = Field(True, env="INCLUDE_DISCLAIMERS")
    disclaimer_text: str = Field(
        "This information is for educational purposes only and should not be considered as financial advice.",
        env="DISCLAIMER_TEXT"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_target_companies_list(self) -> List[str]:
        """Get target companies as a list, handling string input."""
        if isinstance(self.target_companies, str):
            return [company.strip() for company in self.target_companies.split(",")]
        return self.target_companies
    
    def get_supported_years_list(self) -> List[int]:
        """Get supported years as a list, handling string input."""
        if isinstance(self.supported_years, str):
            return [int(year.strip()) for year in self.supported_years.split(",")]
        return self.supported_years

    def validate_llm_config(self) -> bool:
        """Validate LLM configuration based on provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key is not None
        elif self.llm_provider == "azure":
            return all([
                self.azure_openai_endpoint,
                self.azure_openai_key,
                self.azure_openai_deployment_name
            ])
        return False

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for the selected provider."""
        if self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "embedding_model": self.openai_embedding_model
            }
        elif self.llm_provider == "azure":
            return {
                "provider": "azure",
                "endpoint": self.azure_openai_endpoint,
                "api_key": self.azure_openai_key,
                "api_version": self.azure_openai_api_version,
                "deployment_name": self.azure_openai_deployment_name,
                "embedding_deployment": self.azure_openai_embedding_deployment
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")


# Global settings instance
settings = Settings()


# Company-specific configurations
COMPANY_CONFIGS = {
    "castrol": {
        "full_name": "Castrol Limited",
        "parent_company": "BP plc",
        "stock_exchange": "LSE",
        "ticker": "CAS",
        "accounting_standard": "IFRS",
        "fiscal_year_end": "December",
        "currency": "GBP",
        "website": "https://www.castrol.com",
        "investor_relations": "https://www.bp.com/en/global/corporate/investors.html"
    },
    "veedol": {
        "full_name": "Tide Water Oil Company (India) Limited",
        "parent_company": "Independent",
        "stock_exchange": "BSE/NSE",
        "ticker": "TIDEWATER",
        "accounting_standard": "Ind AS",
        "fiscal_year_end": "March",
        "currency": "INR",
        "website": "https://www.veedol.com",
        "investor_relations": "https://www.veedol.com/investor-relations"
    },
    "valvoline": {
        "full_name": "Valvoline Inc.",
        "parent_company": "Independent",
        "stock_exchange": "NYSE",
        "ticker": "VVV",
        "accounting_standard": "US GAAP",
        "fiscal_year_end": "September",
        "currency": "USD",
        "website": "https://www.valvoline.com",
        "investor_relations": "https://investors.valvoline.com"
    }
}


# Financial metrics standardization mapping
FINANCIAL_METRICS_MAPPING = {
    "revenue": ["revenue", "net sales", "total revenue", "sales", "turnover"],
    "operating_income": ["operating income", "operating profit", "EBIT", "earnings before interest and tax"],
    "net_income": ["net income", "net profit", "profit after tax", "PAT"],
    "total_assets": ["total assets", "total asset"],
    "total_equity": ["total equity", "shareholders equity", "stockholders equity"],
    "debt": ["total debt", "borrowings", "long term debt", "short term debt"],
    "cash": ["cash and cash equivalents", "cash", "cash and bank"],
    "operating_margin": ["operating margin", "EBIT margin"],
    "net_margin": ["net margin", "net profit margin", "PAT margin"],
    "roe": ["return on equity", "ROE"],
    "roa": ["return on assets", "ROA"],
    "debt_to_equity": ["debt to equity", "D/E ratio", "debt equity ratio"]
}
