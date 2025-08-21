"""LLM service for financial analysis with specialized prompts and tools."""

from typing import List, Dict, Any, Optional
import structlog
from openai import AsyncOpenAI
import json
from datetime import datetime

from ..data.models.document import QueryResponse, ComparisonResponse

logger = structlog.get_logger()


class FinancialLLM:
    """LLM service specialized for financial analysis and insights."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger.bind(component="financial_llm", model=model)
    
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
            
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self._get_financial_tools(),
                tool_choice="auto"
            )
            
            # Process response
            answer = response.choices[0].message.content
            
            # Handle tool calls if any
            if response.choices[0].message.tool_calls:
                tool_results = await self._handle_tool_calls(
                    response.choices[0].message.tool_calls
                )
                answer += f"\n\nCalculations:\n{tool_results}"
            
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
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self._get_financial_tools(),
                tool_choice="auto"
            )
            
            insights = response.choices[0].message.content
            
            # Handle tool calls for calculations
            if response.choices[0].message.tool_calls:
                tool_results = await self._handle_tool_calls(
                    response.choices[0].message.tool_calls
                )
                insights += f"\n\nCalculations:\n{tool_results}"
            
            return ComparisonResponse(
                comparison_data=context_data,
                insights=insights,
                methodology="Retrieved relevant financial data and performed comparative analysis using LLM reasoning with calculator tools for numerical accuracy.",
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
2. Use the calculator tool for any numerical computations to ensure accuracy
3. Clearly cite sources and indicate confidence levels
4. Distinguish between facts from documents and your analytical interpretations
5. Be precise with financial terminology and accounting concepts
6. Consider different accounting standards (IFRS, US GAAP, Ind AS) when relevant

Guidelines:
- Always base your answers on the provided context documents
- Use exact figures from the documents, not approximations
- If information is not available in the context, clearly state this
- For calculations, use the calculator tool rather than doing math in your head
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
            f"Based on the above context from financial documents, please provide a comprehensive answer to the question. Use the calculator tool for any numerical computations."
        ]
        
        return "\n".join(prompt_parts)
    
    def _build_comparison_system_prompt(self) -> str:
        """Build system prompt for company comparisons."""
        
        return """You are a financial analyst AI assistant specializing in comparative analysis of companies in the lubricants industry (Castrol, Veedol, Valvoline).

Your responsibilities:
1. Perform objective comparative analysis based on provided financial data
2. Use calculator tools for all ratio calculations and growth computations
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
            "Use the calculator tool for all numerical computations and ratios."
        ]
        
        return "\n".join(prompt_parts)
    
    def _get_financial_tools(self) -> List[Dict[str, Any]]:
        """Get available tools for financial calculations."""
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate_financial_ratio",
                    "description": "Calculate financial ratios and perform numerical computations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "The calculation to perform (e.g., 'divide', 'multiply', 'percentage_change', 'cagr')"
                            },
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "The numerical values to use in the calculation"
                            },
                            "description": {
                                "type": "string", 
                                "description": "Description of what is being calculated"
                            }
                        },
                        "required": ["operation", "values", "description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "currency_converter",
                    "description": "Convert between currencies for comparison purposes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number", "description": "Amount to convert"},
                            "from_currency": {"type": "string", "description": "Source currency (USD, GBP, INR, EUR)"},
                            "to_currency": {"type": "string", "description": "Target currency"},
                            "year": {"type": "integer", "description": "Year for historical exchange rate"}
                        },
                        "required": ["amount", "from_currency", "to_currency"]
                    }
                }
            }
        ]
    
    async def _handle_tool_calls(self, tool_calls) -> str:
        """Handle tool calls from the LLM."""
        
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if function_name == "calculate_financial_ratio":
                result = await self._calculate_financial_ratio(**arguments)
                results.append(f"Calculation: {arguments['description']} = {result}")
            
            elif function_name == "currency_converter":
                result = await self._convert_currency(**arguments)
                results.append(f"Currency conversion: {result}")
        
        return "\n".join(results)
    
    async def _calculate_financial_ratio(
        self, 
        operation: str, 
        values: List[float], 
        description: str
    ) -> str:
        """Perform financial calculations."""
        
        try:
            if operation == "divide" and len(values) == 2:
                result = values[0] / values[1] if values[1] != 0 else "undefined"
            elif operation == "multiply" and len(values) >= 2:
                result = 1
                for val in values:
                    result *= val
            elif operation == "percentage_change" and len(values) == 2:
                old_val, new_val = values
                result = ((new_val - old_val) / old_val * 100) if old_val != 0 else "undefined"
            elif operation == "cagr" and len(values) == 3:
                start_val, end_val, years = values
                result = ((end_val / start_val) ** (1/years) - 1) * 100 if start_val != 0 and years != 0 else "undefined"
            else:
                result = "unsupported operation"
            
            return f"{result:.2f}" if isinstance(result, (int, float)) else str(result)
            
        except Exception as e:
            return f"calculation error: {str(e)}"
    
    async def _convert_currency(
        self, 
        amount: float, 
        from_currency: str, 
        to_currency: str,
        year: Optional[int] = None
    ) -> str:
        """Convert currency (placeholder - would need real exchange rate data)."""
        
        # Placeholder exchange rates (would need real-time or historical data)
        exchange_rates = {
            ("USD", "GBP"): 0.79,
            ("USD", "INR"): 83.0,
            ("GBP", "USD"): 1.27,
            ("GBP", "INR"): 105.0,
            ("INR", "USD"): 0.012,
            ("INR", "GBP"): 0.0095,
        }
        
        if from_currency == to_currency:
            return f"{amount:.2f} {to_currency}"
        
        rate = exchange_rates.get((from_currency, to_currency))
        if rate:
            converted = amount * rate
            return f"{amount:.2f} {from_currency} = {converted:.2f} {to_currency}"
        else:
            return f"Exchange rate not available for {from_currency} to {to_currency}"
    
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
    
    async def summarize_document_section(
        self, 
        section_text: str, 
        section_type: str,
        company: str
    ) -> str:
        """Generate a summary of a specific document section."""
        
        system_prompt = f"""You are a financial analyst summarizing a {section_type} section 
        from {company}'s annual report. Provide a concise, factual summary highlighting:
        1. Key financial metrics and figures
        2. Important trends or changes
        3. Management's key messages
        4. Any notable risks or opportunities mentioned
        
        Keep the summary factual and avoid speculation."""
        
        user_prompt = f"Please summarize this {section_type} section:\n\n{section_text}"
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    async def extract_key_metrics(self, text: str, company: str) -> Dict[str, Any]:
        """Extract key financial metrics from text using LLM."""
        
        system_prompt = """You are a financial data extraction specialist. Extract key financial 
        metrics from the provided text and return them in a structured JSON format. 
        
        Focus on:
        - Revenue/Sales figures
        - Profit metrics (gross, operating, net)
        - Balance sheet items (assets, equity, debt)
        - Financial ratios
        - Growth rates
        
        Return only valid JSON with metric names as keys and values as numbers (without currency symbols).
        If a metric is not found, omit it from the response."""
        
        user_prompt = f"Extract key financial metrics from this {company} financial text:\n\n{text}"
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse extracted metrics as JSON")
            return {}
    
    async def validate_financial_statement(self, statement_text: str) -> Dict[str, Any]:
        """Validate financial statement for consistency and completeness."""
        
        system_prompt = """You are a financial auditor AI. Review the provided financial statement 
        text and check for:
        1. Mathematical consistency (do the numbers add up?)
        2. Completeness (are key line items present?)
        3. Unusual patterns or potential errors
        4. Compliance with standard financial reporting formats
        
        Return a JSON response with validation results."""
        
        user_prompt = f"Please validate this financial statement:\n\n{statement_text}"
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"validation_status": "error", "message": "Failed to parse validation results"}
