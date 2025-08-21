# FinSight AI - Financial Knowledge Base

An LLM-powered knowledge base for financial insights on Castrol, Veedol, and Valvoline, built with RAG (Retrieval-Augmented Generation) architecture.

## 🏗️ Architecture Overview

```
Data Ingestion (PDFs/CSVs) → Cleaning & Chunking → Embeddings + Vector DB → RAG Orchestration → LLM (GPT-4/4o) → API/UI → Monitoring & Feedback
```

## 📁 Project Structure

```
finsight-ai/
├── src/
│   ├── data/                    # Data layer
│   │   ├── ingestion/          # PDF/CSV loaders
│   │   ├── storage/            # Blob and vector storage
│   │   └── models/             # Data models
│   ├── preprocessing/           # Text cleaning & chunking
│   ├── embeddings/             # Embedding generation
│   ├── retrieval/              # RAG implementation
│   ├── llm/                    # LLM integration
│   ├── api/                    # REST endpoints
│   └── evaluation/             # RAGAS & feedback
├── tests/                      # Unit and integration tests
├── data/                       # Raw annual reports
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
└── docs/                       # Documentation
```

## 🚀 Features

- **Multi-format Ingestion**: PDF, CSV, XLS annual reports
- **Semantic Search**: Advanced embedding-based retrieval
- **Company Comparison**: Cross-company financial analysis
- **Numerical Accuracy**: Calculator tools for financial ratios
- **Compliance**: Disclaimers and audit trails
- **Scalable**: Generic pipeline for adding new companies

## 🎯 Target Companies

- **Castrol** (BP subsidiary)
- **Veedol** (Tide Water Oil Company)
- **Valvoline** (Independent since 2016)

## 🔧 Technology Stack

- **Backend**: FastAPI
- **LLM**: OpenAI GPT-4/GPT-4o
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector DB**: Chroma (with option for Pinecone/Weaviate)
- **Document Processing**: LangChain, PyPDF2
- **Evaluation**: RAGAS
- **Deployment**: Azure Functions / AWS Lambda

## 📋 Key Constraints Addressed

1. **Data Quality**: OCR error handling, table extraction
2. **Versioning**: Incremental update pipeline
3. **Cross-company Comparisons**: Standardized terminology mapping
4. **Compliance**: Forward-looking statement disclaimers
5. **Context Management**: Multi-year data retrieval with summarization
6. **Numerical Accuracy**: Calculator tools for financial computations
7. **Scalability**: Generic ingestion pipeline

## 🚦 Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env`
3. Run data ingestion: `python scripts/ingest_reports.py`
4. Start API server: `uvicorn src.api.main:app --reload`

## 📊 Usage Examples

```python
# Query financial insights
response = client.post("/query", json={
    "question": "What was Castrol's revenue growth in 2023?",
    "company": "castrol",
    "year": 2023
})

# Compare companies
response = client.post("/compare", json={
    "companies": ["castrol", "valvoline"],
    "metric": "operating_margin",
    "years": [2022, 2023]
})
```

## 🔍 Evaluation Metrics

- **Faithfulness**: Accuracy to source documents
- **Relevance**: Query-response alignment
- **Completeness**: Coverage of financial aspects
- **Numerical Accuracy**: Financial calculation correctness
