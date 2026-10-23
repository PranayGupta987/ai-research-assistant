import os
import sys

# Ensure backend directory is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from app.rag import rag_manager
from app.agents import app_graph

app = FastAPI(title="AI Research Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_docs: List[Dict[str, Any]]
    web_search_results: str
    follow_ups: List[str]

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_count = rag_manager.ingest_pdf(file_path, file.filename)
        return {"filename": file.filename, "chunks_ingested": chunks_count, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_research_assistant(req: QueryRequest):
    try:
        initial_state = {
            "query": req.query,
            "retrieved_docs": [],
            "needs_web_search": False,
            "web_search_results": "",
            "draft_answer": "",
            "fact_check_result": "",
            "final_answer": "",
            "follow_ups": []
        }
        final_state = app_graph.invoke(initial_state)
        return QueryResponse(
            query=final_state["query"],
            final_answer=final_state["final_answer"],
            retrieved_docs=final_state["retrieved_docs"],
            web_search_results=final_state["web_search_results"],
            follow_ups=final_state["follow_ups"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
