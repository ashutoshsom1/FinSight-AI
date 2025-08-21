# Getting Started with FinSight AI

This guide will help you set up and run the FinSight AI financial knowledge base.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Git (for cloning and version control)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd FinSight-AI
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your_api_key_here
   ```

## Quick Start

### 1. Test the System

First, test that everything is working:

```bash
python scripts/test_system.py
```

This will validate that all components can be initialized and basic functionality works.

### 2. Add Sample Data

Place annual reports in the `data/sample_reports/` directory:

```
data/sample_reports/
├── castrol/
│   └── castrol_annual_report_2023.pdf
├── veedol/
│   └── veedol_annual_report_2023.pdf
└── valvoline/
    └── valvoline_annual_report_2023.pdf
```

### 3. Ingest Documents

Run the ingestion pipeline:

```bash
# Ingest all documents from sample_reports directory
python scripts/ingest_reports.py --directory ./data/sample_reports

# Or ingest a single file
python scripts/ingest_reports.py --file ./data/sample_reports/castrol/castrol_annual_report_2023.pdf --company castrol --year 2023
```

### 4. Start the API Server

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Test Queries

Visit `http://localhost:8000/docs` for the interactive API documentation, or use curl:

```bash
# Simple query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What was Castrol'\''s revenue in 2023?"}'

# Company comparison
curl -X POST "http://localhost:8000/compare" \
     -H "Content-Type: application/json" \
     -d '{
       "companies": ["castrol", "valvoline"],
       "metric": "revenue",
       "years": [2022, 2023]
     }'

# Generate summary
curl -X POST "http://localhost:8000/summary?company=castrol&year=2023"
```

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
OPENAI_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
VECTOR_DB_TYPE=chroma
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Company Configuration

Company-specific settings are in `config/settings.py`:

- Accounting standards (IFRS, US GAAP, Ind AS)
- Fiscal year ends
- Currency information
- Stock exchange details

## Usage Examples

### Python API

```python
import asyncio
from src.api.services.financial_service import FinancialService
from src.data.models.document import QueryRequest, Company

async def example_usage():
    # Initialize service (in real app, this is done in FastAPI startup)
    service = await initialize_financial_service()
    
    # Ask a question
    request = QueryRequest(
        question="What was Valvoline's operating margin in 2023?",
        company=Company.VALVOLINE,
        year=2023
    )
    
    response = await service.process_query(request)
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence_score}")
```

### REST API

```javascript
// Query financial data
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "How did Castrol's debt-to-equity ratio change in 2023?",
    company: "castrol",
    year: 2023
  })
});

const result = await response.json();
console.log(result.answer);
```

## Troubleshooting

### Common Issues

1. **"No relevant chunks found"**:
   - Ensure documents are properly ingested
   - Check that your query matches the content in the documents
   - Try broader search terms

2. **OpenAI API errors**:
   - Verify your API key is correct and has sufficient credits
   - Check rate limits if processing many documents

3. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that you're using the correct Python environment

4. **ChromaDB issues**:
   - Delete `./data/chroma_db` directory and re-ingest documents
   - Ensure sufficient disk space

### Debugging

Enable debug logging by setting in `.env`:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

### Performance Optimization

For better performance:

1. **Use smaller chunk sizes** for faster retrieval
2. **Increase batch size** for embedding generation
3. **Consider Pinecone** for production vector storage
4. **Use GPT-3.5-turbo** for faster responses (lower accuracy)

## Next Steps

1. **Add more documents** to improve knowledge coverage
2. **Fine-tune chunking parameters** based on your document types
3. **Implement custom evaluation metrics** for your specific use cases
4. **Add authentication** for production deployment
5. **Set up monitoring** and logging for production use

## Support

For issues and questions:
1. Check the logs for detailed error messages
2. Review the API documentation at `/docs`
3. Run the test suite: `python -m pytest tests/`
