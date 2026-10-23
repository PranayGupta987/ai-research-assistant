import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

class RAGManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_KEY
        )
        self.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    def ingest_pdf(self, file_path: str, filename: str) -> int:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = filename

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        self.vector_store.add_documents(splits)
        return len(splits)

    def query(self, query_text: str) -> List[dict]:
        results = self.vector_store.similarity_search_with_score(query_text, k=4)
        formatted = []
        for doc, score in results:
            formatted.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        return formatted

rag_manager = RAGManager()
