"""Test script to validate the FinSight AI system."""

import asyncio
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.models.document import QueryRequest, ComparisonRequest, Company
from api.services.financial_service import FinancialService
from data.storage.vector_store import create_vector_store
from data.storage.blob_store import create_blob_store
from embeddings.embedding_service import EmbeddingService
from retrieval.rag_engine import RAGEngine
from llm.financial_llm import FinancialLLM
from evaluation.ragas_evaluator import FinancialRAGEvaluator
from config.settings import settings


class SystemTester:
    """Test the complete FinSight AI system."""
    
    def __init__(self):
        self.financial_service = None
        self.evaluator = FinancialRAGEvaluator()
    
    async def initialize(self):
        """Initialize all services for testing."""
        print("Initializing FinSight AI system...")
        
        # Initialize services
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
        
        self.financial_service = FinancialService(
            rag_engine=rag_engine,
            financial_llm=financial_llm,
            blob_store=blob_store
        )
        
        print("System initialization completed!")
    
    async def run_basic_tests(self):
        """Run basic functionality tests."""
        
        print("\n=== Running Basic Tests ===")
        
        # Test 1: Simple query
        print("\n1. Testing simple financial query...")
        try:
            query_request = QueryRequest(
                question="What was Castrol's revenue in 2023?",
                company=Company.CASTROL,
                year=2023
            )
            
            response = await self.financial_service.process_query(query_request)
            print(f"✓ Query processed successfully")
            print(f"  Answer: {response.answer[:200]}...")
            print(f"  Sources: {len(response.sources)}")
            print(f"  Confidence: {response.confidence_score}")
            
        except Exception as e:
            print(f"✗ Query test failed: {e}")
        
        # Test 2: Company comparison
        print("\n2. Testing company comparison...")
        try:
            comparison_request = ComparisonRequest(
                companies=[Company.CASTROL, Company.VALVOLINE],
                metric="revenue",
                years=[2022, 2023]
            )
            
            response = await self.financial_service.compare_companies(comparison_request)
            print(f"✓ Comparison processed successfully")
            print(f"  Insights: {response.insights[:200]}...")
            
        except Exception as e:
            print(f"✗ Comparison test failed: {e}")
        
        # Test 3: Summary generation
        print("\n3. Testing summary generation...")
        try:
            summary = await self.financial_service.generate_summary(
                company="castrol",
                year=2023
            )
            print(f"✓ Summary generated successfully")
            print(f"  Summary: {summary[:200]}...")
            
        except Exception as e:
            print(f"✗ Summary test failed: {e}")
    
    async def run_evaluation_tests(self):
        """Run evaluation framework tests."""
        
        print("\n=== Running Evaluation Tests ===")
        
        # Test RAGAS evaluation
        print("\n1. Testing RAGAS evaluation...")
        try:
            test_query = "What is Valvoline's operating margin?"
            test_response = "Valvoline's operating margin was 15.2% in 2023, showing improvement from 14.8% in 2022."
            test_contexts = [
                "Valvoline reported operating income of $152 million on revenue of $1 billion in 2023.",
                "The company's operating margin improved to 15.2% in 2023 from 14.8% in the previous year."
            ]
            
            evaluation_results = await self.evaluator.evaluate_query_response(
                query=test_query,
                response=test_response,
                retrieved_contexts=test_contexts
            )
            
            print(f"✓ RAGAS evaluation completed")
            print(f"  Overall Score: {evaluation_results.get('overall_score', 'N/A')}")
            print(f"  Faithfulness: {evaluation_results.get('faithfulness', 'N/A')}")
            print(f"  Relevancy: {evaluation_results.get('answer_relevancy', 'N/A')}")
            
        except Exception as e:
            print(f"✗ RAGAS evaluation test failed: {e}")
        
        # Test financial accuracy evaluation
        print("\n2. Testing financial accuracy evaluation...")
        try:
            accuracy_results = await self.evaluator.evaluate_financial_accuracy(
                query=test_query,
                response=test_response,
                source_documents=[
                    {"content": context, "financial_metrics": ["operating_margin"]} 
                    for context in test_contexts
                ]
            )
            
            print(f"✓ Financial accuracy evaluation completed")
            print(f"  Overall Accuracy: {accuracy_results['overall_accuracy']:.2f}")
            
        except Exception as e:
            print(f"✗ Financial accuracy test failed: {e}")
    
    async def run_performance_tests(self):
        """Run performance and load tests."""
        
        print("\n=== Running Performance Tests ===")
        
        # Test embedding generation performance
        print("\n1. Testing embedding generation performance...")
        try:
            test_texts = [
                "Castrol's revenue increased by 12% year-over-year to $15.2 billion.",
                "Operating margin improved to 8.5% from 7.8% in the previous year.",
                "The company reported strong performance in the automotive lubricants segment."
            ]
            
            start_time = asyncio.get_event_loop().time()
            embeddings = await self.financial_service.rag_engine.embedding_service.generate_embeddings(test_texts)
            end_time = asyncio.get_event_loop().time()
            
            print(f"✓ Embedding generation completed")
            print(f"  Time: {end_time - start_time:.2f} seconds")
            print(f"  Embeddings: {len(embeddings)}")
            print(f"  Dimension: {len(embeddings[0]) if embeddings else 0}")
            
        except Exception as e:
            print(f"✗ Embedding performance test failed: {e}")
        
        # Test retrieval performance
        print("\n2. Testing retrieval performance...")
        try:
            start_time = asyncio.get_event_loop().time()
            chunks_with_scores = await self.financial_service.rag_engine.retrieve_relevant_chunks(
                query="revenue growth",
                k=5
            )
            end_time = asyncio.get_event_loop().time()
            
            print(f"✓ Retrieval completed")
            print(f"  Time: {end_time - start_time:.2f} seconds")
            print(f"  Retrieved chunks: {len(chunks_with_scores)}")
            
        except Exception as e:
            print(f"✗ Retrieval performance test failed: {e}")
    
    def print_test_summary(self):
        """Print test summary and next steps."""
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("\nNext Steps:")
        print("1. Add sample financial documents to ./data/ directory")
        print("2. Run: python scripts/ingest_reports.py --directory ./data")
        print("3. Start API server: uvicorn src.api.main:app --reload")
        print("4. Test API endpoints at http://localhost:8000/docs")
        print("\nSample documents needed:")
        print("- Castrol annual reports (PDF)")
        print("- Veedol annual reports (PDF)")  
        print("- Valvoline annual reports (PDF)")
        print("\nConfiguration:")
        print("- Copy .env.example to .env and add your OpenAI API key")
        print("- Adjust settings in config/settings.py as needed")


async def main():
    """Main test function."""
    
    # Setup logging
    from api.middleware.logging import setup_logging
    setup_logging()
    
    tester = SystemTester()
    
    try:
        await tester.initialize()
        await tester.run_basic_tests()
        await tester.run_evaluation_tests()
        await tester.run_performance_tests()
        
    except Exception as e:
        print(f"\nSystem test failed: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")
        
    finally:
        tester.print_test_summary()


if __name__ == "__main__":
    asyncio.run(main())
