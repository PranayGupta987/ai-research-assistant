# AI Research Assistant with RAG & Multi-Agent Orchestration

A production-ready GenAI system where users upload research papers (PDFs) and have deep, multi-agent orchestrated conversations with their corpus.

## Architecture & Features

- **Document Ingestion & RAG**: Parses PDFs (`PyPDF`), chunks text (`RecursiveCharacterTextSplitter`), embeds via OpenAI embeddings, and persists in ChromaDB.
- **Multi-Agent Orchestration (`LangGraph`)**:
  1. **Retrieval Agent**: Pulls relevant document context from ChromaDB.
  2. **Web Search Fallback Agent**: Uses Tavily Search when local corpus context is insufficient.
  3. **Synthesis Agent**: Synthesizes structured answers with source citations.
  4. **Fact-Checking Agent**: Rigorously checks claims against source excerpts to prevent hallucination.
  5. **Follow-up Agent**: Generates intelligent follow-up research questions.
- **Tech Stack**:
  - **Backend**: FastAPI, LangChain, LangGraph, ChromaDB, OpenAI `gpt-4o`, Tavily Search.
  - **Frontend**: Streamlit.

---

## Getting Started

### 1. Environment Setup

Create a `.env` file inside `backend/`:

```env
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
```

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Run Frontend

```bash
cd frontend
pip install streamlit requests
streamlit run app.py
```
