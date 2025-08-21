"""Test script specifically for Azure OpenAI integration."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from embeddings.embedding_factory import create_embedding_service, test_embedding_service
from llm.llm_factory import create_financial_llm, test_financial_llm


async def test_azure_configuration():
    """Test Azure OpenAI configuration."""
    
    print("="*60)
    print("AZURE OPENAI CONFIGURATION TEST")
    print("="*60)
    
    # Check configuration
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Azure Endpoint: {settings.azure_openai_endpoint}")
    print(f"Azure Deployment: {settings.azure_openai_deployment_name}")
    print(f"Azure Embedding Deployment: {settings.azure_openai_embedding_deployment}")
    print(f"API Version: {settings.azure_openai_api_version}")
    
    # Validate configuration
    if not settings.validate_llm_config():
        print("\n❌ Configuration validation failed!")
        print("Please check your Azure OpenAI settings in .env file")
        return False
    
    print("\n✅ Configuration validation passed!")
    return True


async def test_azure_embedding_service():
    """Test Azure OpenAI embedding service."""
    
    print("\n" + "="*60)
    print("AZURE EMBEDDING SERVICE TEST")
    print("="*60)
    
    try:
        # Test embedding service creation
        print("Creating embedding service...")
        embedding_service = create_embedding_service()
        print(f"✅ Embedding service created: {type(embedding_service).__name__}")
        
        # Test connection
        print("\nTesting connection...")
        if await test_embedding_service():
            print("✅ Embedding service connection successful!")
        else:
            print("❌ Embedding service connection failed!")
            return False
        
        # Test single embedding
        print("\nTesting single embedding generation...")
        test_text = "Castrol's revenue increased by 12% in 2023."
        embedding = await embedding_service.generate_single_embedding(test_text)
        
        print(f"✅ Generated embedding with dimension: {len(embedding)}")
        print(f"   Sample values: {embedding[:5]}...")
        
        # Test batch embeddings
        print("\nTesting batch embedding generation...")
        test_texts = [
            "Revenue growth was strong in Q4 2023.",
            "Operating margins improved year-over-year.",
            "The company reported record profits."
        ]
        
        embeddings = await embedding_service.generate_embeddings(test_texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False


async def test_azure_llm_service():
    """Test Azure OpenAI LLM service."""
    
    print("\n" + "="*60)
    print("AZURE LLM SERVICE TEST")
    print("="*60)
    
    try:
        # Test LLM service creation
        print("Creating LLM service...")
        llm_service = create_financial_llm()
        print(f"✅ LLM service created: {type(llm_service).__name__}")
        
        # Test connection
        print("\nTesting connection...")
        if await test_financial_llm():
            print("✅ LLM service connection successful!")
        else:
            print("❌ LLM service connection failed!")
            return False
        
        # Test simple query
        print("\nTesting simple financial query...")
        
        # Create mock context for testing
        mock_context = {
            "formatted_context": "Castrol reported revenue of $15.2 billion in 2023, up from $13.6 billion in 2022.",
            "sources": [{"chunk_id": "test_chunk", "similarity_score": 0.95}],
            "chunks": [],
            "chunk_count": 1
        }
        
        response = await llm_service.generate_financial_response(
            query="What was Castrol's revenue in 2023?",
            context=mock_context,
            include_disclaimer=True
        )
        
        print(f"✅ Generated response:")
        print(f"   Answer: {response.answer[:200]}...")
        print(f"   Confidence: {response.confidence_score}")
        print(f"   Processing time: {response.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        return False


async def test_end_to_end_workflow():
    """Test end-to-end workflow with Azure OpenAI."""
    
    print("\n" + "="*60)
    print("END-TO-END WORKFLOW TEST")
    print("="*60)
    
    try:
        # Create services
        embedding_service = create_embedding_service()
        llm_service = create_financial_llm()
        
        # Test document processing workflow
        print("Testing document processing workflow...")
        
        # Sample financial text
        sample_text = """
        Castrol Financial Performance 2023
        
        Revenue: $15.2 billion (up 12% from 2022)
        Operating Income: $1.3 billion 
        Net Income: $950 million
        Operating Margin: 8.5%
        
        The company delivered strong performance across all segments,
        with particular strength in automotive lubricants.
        """
        
        # Generate embedding
        print("1. Generating embedding for sample text...")
        embedding = await embedding_service.generate_single_embedding(sample_text)
        print(f"   ✅ Embedding generated (dimension: {len(embedding)})")
        
        # Simulate retrieval context
        print("2. Creating retrieval context...")
        context = {
            "formatted_context": sample_text,
            "sources": [{"chunk_id": "sample_chunk", "similarity_score": 0.98}],
            "chunks": [],
            "chunk_count": 1
        }
        
        # Generate response
        print("3. Generating LLM response...")
        response = await llm_service.generate_financial_response(
            query="What was Castrol's operating margin in 2023?",
            context=context
        )
        
        print(f"   ✅ Response generated:")
        print(f"   Answer: {response.answer}")
        print(f"   Confidence: {response.confidence_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        return False


async def main():
    """Run all Azure OpenAI tests."""
    
    print("Azure OpenAI Integration Test Suite")
    print("This will test your Azure OpenAI configuration and services.")
    
    tests = [
        ("Configuration", test_azure_configuration),
        ("Embedding Service", test_azure_embedding_service),
        ("LLM Service", test_azure_llm_service),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if await test_func():
                passed_tests += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with error: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Azure OpenAI integration is working correctly.")
        print("\nNext steps:")
        print("1. Add sample documents to data/sample_reports/")
        print("2. Run: python scripts/ingest_reports.py --directory ./data/sample_reports")
        print("3. Run: uvicorn src.api.main:app --reload")
        print("4. Visit: http://localhost:8000/docs")
        
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed.")
        print("\nTroubleshooting:")
        print("1. Verify your Azure OpenAI endpoint and API key")
        print("2. Check that your deployment names are correct")
        print("3. Ensure you have sufficient quota in Azure OpenAI")
        print("4. Verify network connectivity to Azure")
        
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
