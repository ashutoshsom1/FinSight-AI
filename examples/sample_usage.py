"""Sample usage examples for FinSight AI."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.models.document import QueryRequest, ComparisonRequest, Company
from api.services.financial_service import FinancialService
from data.storage.vector_store import create_vector_store
from data.storage.blob_store import create_blob_store
from embeddings.embedding_service import EmbeddingService
from retrieval.rag_engine import RAGEngine
from llm.financial_llm import FinancialLLM
from config.settings import settings


async def initialize_services():
    """Initialize all services for examples."""
    
    print("Initializing FinSight AI services...")
    
    # Initialize core services
    vector_store = await create_vector_store(
        settings.vector_db_type,
        persist_directory=settings.chroma_persist_directory
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
    
    print("✓ Services initialized successfully")
    return financial_service


async def example_basic_queries(service: FinancialService):
    """Example basic financial queries."""
    
    print("\n" + "="*50)
    print("BASIC QUERY EXAMPLES")
    print("="*50)
    
    queries = [
        {
            "question": "What was Castrol's revenue in 2023?",
            "company": Company.CASTROL,
            "year": 2023
        },
        {
            "question": "How did Valvoline's operating margin change over the last 3 years?",
            "company": Company.VALVOLINE
        },
        {
            "question": "What are the main risk factors for Veedol?",
            "company": Company.VEEDOL
        },
        {
            "question": "Compare debt-to-equity ratios across all three companies"
        }
    ]
    
    for i, query_data in enumerate(queries, 1):
        print(f"\n{i}. Query: {query_data['question']}")
        
        try:
            request = QueryRequest(**query_data)
            response = await service.process_query(request)
            
            print(f"   Answer: {response.answer[:200]}...")
            print(f"   Confidence: {response.confidence_score:.2f}")
            print(f"   Sources: {len(response.sources)}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")


async def example_company_comparisons(service: FinancialService):
    """Example company comparison queries."""
    
    print("\n" + "="*50)
    print("COMPANY COMPARISON EXAMPLES")
    print("="*50)
    
    comparisons = [
        {
            "companies": [Company.CASTROL, Company.VALVOLINE],
            "metric": "revenue",
            "years": [2022, 2023]
        },
        {
            "companies": [Company.CASTROL, Company.VEEDOL, Company.VALVOLINE],
            "metric": "operating_margin",
            "years": [2023]
        },
        {
            "companies": [Company.VALVOLINE, Company.VEEDOL],
            "metric": "debt_to_equity",
            "years": [2021, 2022, 2023]
        }
    ]
    
    for i, comparison_data in enumerate(comparisons, 1):
        companies_str = ", ".join([c.value for c in comparison_data["companies"]])
        print(f"\n{i}. Compare {companies_str} on {comparison_data['metric']}")
        
        try:
            request = ComparisonRequest(**comparison_data)
            response = await service.compare_companies(request)
            
            print(f"   Insights: {response.insights[:300]}...")
            print(f"   Data points: {len(response.sources)}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")


async def example_summaries(service: FinancialService):
    """Example summary generation."""
    
    print("\n" + "="*50)
    print("SUMMARY GENERATION EXAMPLES")
    print("="*50)
    
    summaries = [
        {"company": "castrol", "year": 2023},
        {"company": "valvoline", "year": 2023, "section_type": "income_statement"},
        {"company": "veedol", "year": 2022}
    ]
    
    for i, summary_data in enumerate(summaries, 1):
        company = summary_data["company"]
        year = summary_data["year"]
        section = summary_data.get("section_type", "overall")
        
        print(f"\n{i}. {company.title()} {year} {section} summary")
        
        try:
            summary = await service.generate_summary(**summary_data)
            print(f"   Summary: {summary[:300]}...")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")


async def example_advanced_queries(service: FinancialService):
    """Example advanced analytical queries."""
    
    print("\n" + "="*50)
    print("ADVANCED ANALYSIS EXAMPLES")
    print("="*50)
    
    advanced_queries = [
        "What factors contributed to Castrol's revenue growth in 2023?",
        "How do the three companies compare in terms of sustainability initiatives?",
        "What are the key differences in business models between Castrol, Veedol, and Valvoline?",
        "Which company has the strongest balance sheet and why?",
        "What are the main competitive advantages of each company?",
        "How have raw material costs affected profitability across the companies?",
        "What geographic markets are most important for each company?",
        "How do the companies' R&D investments compare?"
    ]
    
    for i, question in enumerate(advanced_queries, 1):
        print(f"\n{i}. {question}")
        
        try:
            request = QueryRequest(question=question, max_chunks=15)
            response = await service.process_query(request)
            
            print(f"   Answer: {response.answer[:250]}...")
            print(f"   Confidence: {response.confidence_score:.2f}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")


async def demonstrate_evaluation(service: FinancialService):
    """Demonstrate evaluation capabilities."""
    
    print("\n" + "="*50)
    print("EVALUATION DEMONSTRATION")
    print("="*50)
    
    from evaluation.ragas_evaluator import FinancialRAGEvaluator
    
    evaluator = FinancialRAGEvaluator()
    
    # Example evaluation
    test_query = "What was Valvoline's revenue in 2023?"
    
    try:
        # Get response from system
        request = QueryRequest(question=test_query, company=Company.VALVOLINE, year=2023)
        response = await service.process_query(request)
        
        # Evaluate the response
        contexts = [chunk["content"] for chunk in response.retrieved_chunks]
        
        evaluation_results = await evaluator.evaluate_query_response(
            query=test_query,
            response=response.answer,
            retrieved_contexts=contexts
        )
        
        print(f"Query: {test_query}")
        print(f"Response: {response.answer[:200]}...")
        print(f"\nEvaluation Results:")
        for metric, score in evaluation_results.items():
            if isinstance(score, (int, float)):
                print(f"  {metric}: {score:.3f}")
        
        # Financial accuracy evaluation
        accuracy_results = await evaluator.evaluate_financial_accuracy(
            query=test_query,
            response=response.answer,
            source_documents=[
                {"content": chunk["content"], "financial_metrics": chunk["financial_metrics"]}
                for chunk in response.retrieved_chunks
            ]
        )
        
        print(f"\nFinancial Accuracy: {accuracy_results['overall_accuracy']:.3f}")
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")


async def main():
    """Run all examples."""
    
    print("FinSight AI - Sample Usage Examples")
    print("Make sure you have:")
    print("1. Set OPENAI_API_KEY in .env")
    print("2. Ingested some sample documents")
    print("3. Installed all dependencies")
    
    try:
        # Initialize services
        service = await initialize_services()
        
        # Run examples
        await example_basic_queries(service)
        await example_company_comparisons(service)
        await example_summaries(service)
        await example_advanced_queries(service)
        await demonstrate_evaluation(service)
        
        print("\n" + "="*50)
        print("EXAMPLES COMPLETED")
        print("="*50)
        print("\nTo run the API server:")
        print("uvicorn src.api.main:app --reload")
        print("\nThen visit: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"\n✗ Examples failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure OpenAI API key is set in .env")
        print("2. Run: python scripts/ingest_reports.py --directory ./data/sample_reports")
        print("3. Check that sample documents exist in data/sample_reports/")


if __name__ == "__main__":
    asyncio.run(main())
