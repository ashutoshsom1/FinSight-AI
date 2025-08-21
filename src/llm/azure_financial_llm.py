"""Azure OpenAI LLM service for financial analysis with specialized prompts and tools."""

from typing import List, Dict, Any, Optional
import structlog
import json
from datetime import datetime

from ..data.models.document import QueryResponse, ComparisonResponse

logger = structlog.get_logger()


class AzureFinancialLLM:
    """LLM service specialized for financial analysis using Azure OpenAI."""
    
    def __init__(
        self, 
        endpoint: str,
        api_key: str,
        api_version: str = "2024-10-21",
        deployment_name: str = "gpt-35-turbo-16k",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger.bind(
            component="azure_financial_llm", 
            deployment=deployment_name,
            endpoint=endpoint
        )
        
        # Initialize Azure OpenAI client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            from openai import AsyncAzureOpenAI
            
            self.client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
            
            self.logger.info("Azure OpenAI client initialized successfully")
            
        except ImportError:
            raise ImportError(
                "Azure OpenAI client not available. Install with: pip install openai[azure]"
            )
        except Exception as e:
            self.logger.error("Failed to initialize Azure OpenAI client", error=str(e))
            raise
    
    async def generate_financial_response(
        self, 
        query: str,
        context: Dict[str, Any],
        include_disclaimer: bool = True
    ) -> QueryResponse:
        """Generate a response to a financial query using retrieved context."""
        
        self.logger.info("Generating financial response", query=query[:100])
        
        start_time = datetime.now()
        
        try:
            # Build the prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context)
            
            # Call Azure OpenAI
            response = await self.client.chat.completions.create(
                model=self.deployment_name,  # This is the deployment name in Azure
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Process response
            answer = response.choices[0].message.content
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build response
            query_response = QueryResponse(
                answer=answer,
                sources=context["sources"],
                confidence_score=self._calculate_confidence_score(context),
                disclaimer=self._get_disclaimer() if include_disclaimer else None,
                processing_time=processing_time,
                retrieved_chunks=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        "section_type": chunk.section_type,
                        "financial_metrics": chunk.financial_metrics
                    }
                    for chunk in context["chunks"]
                ]
            )
            
            self.logger.info(
                "Financial response generated", 
                processing_time=processing_time,
                answer_length=len(answer)
            )
            
            return query_response
            
        except Exception as e:
            self.logger.error("Failed to generate financial response", error=str(e))
            raise
    
    async def generate_comparison_response(
        self, 
        companies: List[str],
        metric: str,
        context_data: Dict[str, Any]
    ) -> ComparisonResponse:
        """Generate a comparison response between companies."""
        
        self.logger.info("Generating comparison response", companies=companies, metric=metric)
        
        try:
            system_prompt = self._build_comparison_system_prompt()
            user_prompt = self._build_comparison_user_prompt(companies, metric, context_data)
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            insights = response.choices[0].message.content
            
            return ComparisonResponse(
                comparison_data=context_data,
                insights=insights,
                methodology="Retrieved relevant financial data and performed comparative analysis using Azure OpenAI reasoning.",
                disclaimers=[self._get_disclaimer()],
                sources=context_data.get("sources", [])
            )
            
        except Exception as e:
            self.logger.error("Failed to generate comparison response", error=str(e))
            raise
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for financial analysis."""
        
        return """You are a financial analyst AI assistant specializing in analyzing annual reports and financial data for Castrol, Veedol, and Valvoline. 

Your responsibilities:
1. Provide accurate, data-driven financial insights based on the provided context
2. Perform numerical computations carefully and show your work
3. Clearly cite sources and indicate confidence levels
4. Distinguish between facts from documents and your analytical interpretations
5. Be precise with financial terminology and accounting concepts
6. Consider different accounting standards (IFRS, US GAAP, Ind AS) when relevant

Guidelines:
- Always base your answers on the provided context documents
- Use exact figures from the documents, not approximations
- If information is not available in the context, clearly state this
- For calculations, show your work step by step
- Consider currency differences when comparing companies
- Be aware of different fiscal year ends across companies

Response format:
- Start with a direct answer to the question
- Provide supporting details and context
- Include relevant financial metrics and calculations
- End with source citations and confidence assessment"""
    
    def _build_user_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build user prompt with query and context."""
        
        prompt_parts = [
            f"Question: {query}",
            "",
            "Relevant Financial Document Context:",
            context["formatted_context"],
            "",
            f"Based on the above context from financial documents, please provide a comprehensive answer to the question. Show any numerical calculations step by step."
        ]
        
        return "\n".join(prompt_parts)
    
    def _build_comparison_system_prompt(self) -> str:
        """Build system prompt for company comparisons."""
        
        return """You are a financial analyst AI assistant specializing in comparative analysis of companies in the lubricants industry (Castrol, Veedol, Valvoline).

Your responsibilities:
1. Perform objective comparative analysis based on provided financial data
2. Show all ratio calculations and growth computations step by step
3. Account for different accounting standards and currencies
4. Identify trends, strengths, and areas of concern for each company
5. Provide actionable insights while maintaining objectivity

Key considerations:
- Castrol (BP subsidiary): Reports in GBP, follows IFRS, December year-end
- Veedol (India): Reports in INR, follows Ind AS, March year-end  
- Valvoline (US): Reports in USD, follows US GAAP, September year-end

Always normalize for currency and accounting differences when making comparisons."""
    
    def _build_comparison_user_prompt(
        self, 
        companies: List[str], 
        metric: str, 
        context_data: Dict[str, Any]
    ) -> str:
        """Build user prompt for company comparison."""
        
        prompt_parts = [
            f"Compare {', '.join(companies)} on the metric: {metric}",
            "",
            "Financial Data Context:",
            json.dumps(context_data, indent=2),
            "",
            "Please provide a comprehensive comparison including:",
            "1. Current performance on the specified metric",
            "2. Trends over time",
            "3. Relative positioning among the companies",
            "4. Key insights and implications",
            "5. Any notable differences in accounting or reporting that affect comparability",
            "",
            "Show all numerical computations and ratios step by step."
        ]
        
        return "\n".join(prompt_parts)
    
    def _calculate_confidence_score(self, context: Dict[str, Any]) -> float:
        """Calculate confidence score based on context quality."""
        
        # Base confidence on number and quality of sources
        chunk_count = context["chunk_count"]
        avg_similarity = sum(
            source["similarity_score"] for source in context["sources"]
        ) / len(context["sources"]) if context["sources"] else 0
        
        # Confidence factors
        chunk_factor = min(chunk_count / 5, 1.0)  # Optimal around 5 chunks
        similarity_factor = avg_similarity
        
        confidence = (chunk_factor + similarity_factor) / 2
        return min(confidence, 0.95)  # Cap at 95%
    
    def _get_disclaimer(self) -> str:
        """Get standard financial disclaimer."""
        
        return ("This analysis is based on publicly available financial documents and is for "
                "informational purposes only. It should not be considered as investment advice, "
                "financial advice, or a recommendation to buy or sell securities. Please consult "
                "with qualified financial professionals before making investment decisions. "
                "Past performance does not guarantee future results.")
    
    async def test_connection(self) -> bool:
        """Test the Azure OpenAI connection."""
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return len(response.choices) > 0
        except Exception as e:
            self.logger.error("Azure OpenAI connection test failed", error=str(e))
            return False
