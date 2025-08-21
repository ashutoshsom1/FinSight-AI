# Azure OpenAI Setup Guide

This guide explains how to configure FinSight AI to use Azure OpenAI instead of standard OpenAI.

## Prerequisites

1. **Azure OpenAI Resource**: You need an Azure OpenAI resource with deployed models
2. **API Access**: Ensure you have the endpoint URL and API key
3. **Model Deployments**: Deploy the required models (GPT and embedding models)

## Configuration

### 1. Environment Variables

Update your `.env` file with Azure OpenAI settings:

```bash
# Set provider to Azure
LLM_PROVIDER=azure

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo-16k
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

### 2. Your Current Configuration

Based on your provided credentials, your `.env` file should contain:

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://openaitest-005.openai.azure.com/
AZURE_OPENAI_KEY=dfc1185767ac474c9be05d2ac537f410
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-35-turbo-16k
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## Required Model Deployments

Ensure you have deployed these models in your Azure OpenAI resource:

### 1. **Chat Completion Model**
- **Model**: GPT-3.5-turbo-16k or GPT-4
- **Deployment Name**: `gpt-35-turbo-16k` (as configured above)
- **Purpose**: Financial analysis and response generation

### 2. **Embedding Model**
- **Model**: text-embedding-ada-002
- **Deployment Name**: `text-embedding-ada-002` (as configured above)
- **Purpose**: Document embedding and semantic search

## Testing Your Setup

### 1. Quick Test

Run the Azure OpenAI specific test:

```bash
python scripts/test_azure_openai.py
```

This will test:
- Configuration validation
- Embedding service connection
- LLM service connection
- End-to-end workflow

### 2. Full System Validation

Run the complete validation:

```bash
python scripts/validate_setup.py
```

### 3. Manual Testing

You can also test manually:

```python
import asyncio
from src.embeddings.embedding_factory import create_embedding_service
from src.llm.llm_factory import create_financial_llm

async def test_azure():
    # Test embedding
    embedding_service = create_embedding_service()
    embedding = await embedding_service.generate_single_embedding("test")
    print(f"Embedding dimension: {len(embedding)}")
    
    # Test LLM
    llm_service = create_financial_llm()
    # Test with mock context...

asyncio.run(test_azure())
```

## Differences from Standard OpenAI

### 1. **API Structure**
- Azure OpenAI uses deployment names instead of model names
- Different endpoint structure
- API versioning is explicit

### 2. **Rate Limits**
- Azure OpenAI has different rate limiting
- Quota management through Azure portal

### 3. **Model Availability**
- Model availability depends on your Azure region
- Some models may not be available in all regions

## Troubleshooting

### Common Issues

1. **"Deployment not found" error**
   - Verify your deployment names in Azure OpenAI Studio
   - Ensure models are successfully deployed

2. **"Invalid API key" error**
   - Check your API key in Azure portal
   - Ensure the key has proper permissions

3. **"Quota exceeded" error**
   - Check your quota limits in Azure portal
   - Consider upgrading your pricing tier

4. **"Model not available" error**
   - Verify the model is deployed in your resource
   - Check the deployment name matches your configuration

### Debugging Steps

1. **Check Azure OpenAI Studio**
   - Verify your deployments are active
   - Test the models directly in the playground

2. **Verify Network Connectivity**
   ```bash
   curl -H "api-key: YOUR_API_KEY" \
        "https://your-resource.openai.azure.com/openai/deployments?api-version=2024-10-21"
   ```

3. **Check Logs**
   - Enable debug logging: `DEBUG=true` in `.env`
   - Check application logs for detailed error messages

## Performance Considerations

### 1. **Regional Deployment**
- Deploy Azure OpenAI in the same region as your application
- Consider latency implications

### 2. **Quota Management**
- Monitor your token usage
- Set up alerts for quota limits

### 3. **Cost Optimization**
- Use appropriate model sizes for your use case
- Consider caching for repeated queries

## Security Best Practices

1. **API Key Management**
   - Store API keys securely (Azure Key Vault recommended)
   - Rotate keys regularly
   - Use managed identities when possible

2. **Network Security**
   - Configure network access restrictions
   - Use private endpoints for production

3. **Monitoring**
   - Enable Azure Monitor for API usage
   - Set up alerts for unusual activity

## Migration from Standard OpenAI

If migrating from standard OpenAI:

1. **Update Configuration**
   - Change `LLM_PROVIDER` to `azure`
   - Add Azure OpenAI settings

2. **Deploy Models**
   - Deploy equivalent models in Azure OpenAI
   - Update deployment names in configuration

3. **Test Thoroughly**
   - Run all test suites
   - Verify response quality matches expectations

4. **Monitor Performance**
   - Compare latency and throughput
   - Adjust configuration as needed

## Next Steps

Once Azure OpenAI is configured:

1. **Add Sample Data**
   ```bash
   # Add PDFs to data/sample_reports/
   python scripts/ingest_reports.py --directory ./data/sample_reports
   ```

2. **Start the API**
   ```bash
   uvicorn src.api.main:app --reload
   ```

3. **Test Queries**
   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "What was Castrol'\''s revenue in 2023?"}'
   ```

Your FinSight AI system is now configured to use Azure OpenAI! 🚀
