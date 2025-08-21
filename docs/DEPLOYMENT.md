# Deployment Guide

This guide covers deploying FinSight AI to production environments.

## Deployment Options

### 1. Local Development
- Use for development and testing
- SQLite-based ChromaDB
- Local file storage

### 2. Cloud Deployment (Recommended)
- Azure Functions / AWS Lambda for serverless
- Pinecone for vector storage
- Azure Blob Storage / AWS S3 for documents

### 3. Container Deployment
- Docker containers
- Kubernetes orchestration
- Scalable and portable

## Environment Configuration

### Production Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_production_api_key
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Vector Database (Production)
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=finsight-production

# Storage (Production)
BLOB_STORAGE_TYPE=azure
AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string

# Application
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Security
INCLUDE_DISCLAIMERS=true
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p data/chroma_db data/documents data/feedback logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  finsight-ai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_DB_TYPE=chroma
      - BLOB_STORAGE_TYPE=local
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    
  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

## Azure Functions Deployment

### Function App Structure

```
finsight-function/
├── requirements.txt
├── host.json
├── function_app.py
└── query/
    ├── __init__.py
    └── function.json
```

### function_app.py

```python
import azure.functions as func
import logging
import json
from src.api.services.financial_service import FinancialService

app = func.FunctionApp()

# Initialize service (would need proper initialization)
financial_service = None

@app.route(route="query", methods=["POST"])
async def query_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
        # Process query using financial_service
        # Return response
        pass
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
```

## AWS Lambda Deployment

### Serverless Framework

```yaml
# serverless.yml
service: finsight-ai

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  environment:
    OPENAI_API_KEY: ${env:OPENAI_API_KEY}
    VECTOR_DB_TYPE: pinecone
    PINECONE_API_KEY: ${env:PINECONE_API_KEY}

functions:
  query:
    handler: src.api.lambda_handler.query_handler
    events:
      - http:
          path: query
          method: post
          cors: true
    timeout: 30
    memorySize: 1024

  compare:
    handler: src.api.lambda_handler.compare_handler
    events:
      - http:
          path: compare
          method: post
          cors: true

plugins:
  - serverless-python-requirements
```

## Production Considerations

### 1. Security

- **API Authentication**: Implement JWT or API key authentication
- **Rate Limiting**: Protect against abuse
- **Input Validation**: Sanitize all inputs
- **HTTPS**: Use SSL/TLS in production

### 2. Monitoring

- **Application Metrics**: Response times, error rates
- **Business Metrics**: Query types, user satisfaction
- **Infrastructure Metrics**: CPU, memory, storage usage
- **Cost Monitoring**: OpenAI API usage, cloud resources

### 3. Scalability

- **Horizontal Scaling**: Multiple API instances
- **Caching**: Cache frequent queries and embeddings
- **Database Optimization**: Proper indexing and partitioning
- **CDN**: For static assets and documentation

### 4. Data Management

- **Backup Strategy**: Regular backups of vector database
- **Version Control**: Track document versions and updates
- **Data Retention**: Policies for old documents and logs
- **Compliance**: Ensure data handling meets regulatory requirements

### 5. Performance Optimization

- **Embedding Caching**: Cache embeddings for repeated content
- **Query Optimization**: Optimize vector search parameters
- **Batch Processing**: Process multiple documents efficiently
- **Connection Pooling**: Optimize database connections

## Monitoring Setup

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
query_counter = Counter('finsight_queries_total', 'Total queries processed')
query_duration = Histogram('finsight_query_duration_seconds', 'Query processing time')
active_documents = Gauge('finsight_active_documents', 'Number of active documents')
```

### Health Checks

```python
@app.get("/health/detailed")
async def detailed_health_check():
    checks = {
        "vector_store": await check_vector_store_health(),
        "llm_service": await check_llm_service_health(),
        "blob_storage": await check_blob_storage_health(),
    }
    
    overall_healthy = all(checks.values())
    
    return {
        "status": "healthy" if overall_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }
```

## Cost Optimization

### 1. OpenAI API Costs

- **Model Selection**: Use GPT-3.5-turbo for simpler queries
- **Prompt Optimization**: Minimize token usage
- **Caching**: Cache responses for identical queries
- **Batch Processing**: Process multiple embeddings together

### 2. Infrastructure Costs

- **Auto-scaling**: Scale down during low usage
- **Reserved Instances**: For predictable workloads
- **Storage Optimization**: Compress documents, clean old data
- **CDN Usage**: Cache static content

## Security Checklist

- [ ] API authentication implemented
- [ ] Input validation and sanitization
- [ ] Rate limiting configured
- [ ] HTTPS/SSL enabled
- [ ] Environment variables secured
- [ ] Database access restricted
- [ ] Logging configured (no sensitive data)
- [ ] Error handling (no information leakage)
- [ ] Regular security updates
- [ ] Backup and disaster recovery plan

## Deployment Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Database migrations run
- [ ] Sample data ingested
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Logging configured
- [ ] Backup strategy implemented
- [ ] Documentation updated
- [ ] Team training completed
