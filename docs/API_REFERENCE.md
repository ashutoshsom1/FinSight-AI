# API Reference

FinSight AI provides a REST API for querying financial data and generating insights.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for local development. For production deployment, implement appropriate authentication mechanisms.

## Endpoints

### GET /

Get API information and available endpoints.

**Response:**
```json
{
  "name": "FinSight AI",
  "version": "0.1.0",
  "description": "LLM-powered knowledge base for financial insights",
  "companies": ["castrol", "veedol", "valvoline"],
  "supported_years": [2020, 2021, 2022, 2023, 2024],
  "endpoints": {
    "query": "/query - Ask financial questions",
    "compare": "/compare - Compare companies",
    "summary": "/summary - Generate summaries",
    "health": "/health - Health check"
  }
}
```

### GET /health

Health check endpoint to verify system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "services": ["financial_service", "vector_store", "embedding_service"]
}
```

### POST /query

Ask financial questions and get AI-powered insights.

**Request Body:**
```json
{
  "question": "What was Castrol's revenue growth in 2023?",
  "company": "castrol",  // Optional: castrol, veedol, valvoline
  "year": 2023,          // Optional: filter by year
  "section_type": "income_statement",  // Optional: filter by section
  "max_chunks": 10,      // Optional: max chunks to retrieve (1-50)
  "include_context": true // Optional: include retrieved chunks in response
}
```

**Response:**
```json
{
  "answer": "Castrol's revenue increased by 12% in 2023, reaching $15.2 billion compared to $13.6 billion in 2022...",
  "sources": [
    {
      "chunk_id": "chunk_123",
      "document_id": "doc_456", 
      "section_type": "income_statement",
      "similarity_score": 0.92
    }
  ],
  "confidence_score": 0.88,
  "disclaimer": "This information is for educational purposes only...",
  "processing_time": 2.34,
  "retrieved_chunks": [
    {
      "chunk_id": "chunk_123",
      "content": "Revenue increased to $15.2 billion...",
      "section_type": "income_statement",
      "financial_metrics": ["revenue"]
    }
  ]
}
```

### POST /compare

Compare financial metrics across companies.

**Request Body:**
```json
{
  "companies": ["castrol", "valvoline"],
  "metric": "operating_margin",
  "years": [2022, 2023],
  "normalize_currency": true  // Optional: convert to common currency
}
```

**Response:**
```json
{
  "comparison_data": {
    "castrol": {
      "2022": [{"value": 8.2, "source_chunk": "chunk_123"}],
      "2023": [{"value": 8.7, "source_chunk": "chunk_124"}]
    },
    "valvoline": {
      "2022": [{"value": 15.1, "source_chunk": "chunk_456"}],
      "2023": [{"value": 15.8, "source_chunk": "chunk_457"}]
    }
  },
  "insights": "Valvoline maintains significantly higher operating margins than Castrol...",
  "methodology": "Retrieved relevant financial data and performed comparative analysis...",
  "disclaimers": ["This information is for educational purposes only..."],
  "sources": [...]
}
```

### POST /summary

Generate summary of company's financial performance.

**Query Parameters:**
- `company` (required): Company name (castrol, veedol, valvoline)
- `year` (required): Year to summarize
- `section_type` (optional): Specific section to summarize

**Example:**
```
POST /summary?company=castrol&year=2023&section_type=income_statement
```

**Response:**
```json
{
  "summary": "Castrol delivered strong financial performance in 2023 with revenue growth of 12% to $15.2 billion..."
}
```

### POST /ingest

Ingest a new financial document (background processing).

**Request Body:**
```json
{
  "file_path": "./data/new_report.pdf",
  "company": "castrol",
  "year": 2024,
  "document_type": "annual_report"  // Optional
}
```

**Response:**
```json
{
  "message": "Document ingestion started",
  "file_path": "./data/new_report.pdf",
  "company": "castrol",
  "year": 2024,
  "status": "processing"
}
```

### GET /documents

List available documents in the knowledge base.

**Query Parameters:**
- `company` (optional): Filter by company
- `year` (optional): Filter by year

**Response:**
```json
{
  "documents": [
    {
      "name": "castrol/2023/castrol_annual_report_2023.pdf",
      "size": 2048576,
      "modified": "2024-01-15T10:00:00",
      "metadata": {
        "company": "castrol",
        "year": 2023,
        "document_type": "annual_report"
      }
    }
  ]
}
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error description"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (document/data not found)
- `500`: Internal Server Error (system error)
- `503`: Service Unavailable (system not ready)

## Rate Limits

Currently no rate limits are enforced. For production deployment, implement appropriate rate limiting based on your OpenAI API limits.

## Data Models

### QueryRequest
```json
{
  "question": "string (required)",
  "company": "castrol|veedol|valvoline (optional)",
  "year": "integer (optional)",
  "section_type": "string (optional)",
  "max_chunks": "integer 1-50 (optional, default: 10)",
  "include_context": "boolean (optional, default: true)"
}
```

### ComparisonRequest
```json
{
  "companies": ["array of company names (required)"],
  "metric": "string (required)",
  "years": ["array of integers (required)"],
  "normalize_currency": "boolean (optional, default: true)"
}
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI, where you can:

- Explore all endpoints
- Test API calls directly in the browser
- View detailed request/response schemas
- Download OpenAPI specification
