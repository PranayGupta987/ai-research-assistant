import os
from typing import List, Dict, Any, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from app.rag import rag_manager
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    needs_web_search: bool
    web_search_results: str
    draft_answer: str
    fact_check_result: str
    final_answer: str
    follow_ups: List[str]

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2, google_api_key=GEMINI_KEY)
tavily_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_KEY)

def retrieval_node(state: AgentState) -> AgentState:
    query = state["query"]
    docs = rag_manager.query(query)
    needs_web = False
    if not docs or all(d.get("score", 1.0) > 0.8 for d in docs):
        needs_web = True

    return {
        **state,
        "retrieved_docs": docs,
        "needs_web_search": needs_web
    }

def web_search_node(state: AgentState) -> AgentState:
    if not state.get("needs_web_search", False):
        return {**state, "web_search_results": "Web search skipped (local corpus sufficient)."}

    query = state["query"]
    try:
        results = tavily_tool.invoke(query)
        formatted = "\n".join([f"- {r.get('content', '')} ({r.get('url', '')})" for r in results])
        return {**state, "web_search_results": formatted}
    except Exception as e:
        return {**state, "web_search_results": f"Web search failed: {str(e)}"}

def synthesis_node(state: AgentState) -> AgentState:
    query = state["query"]
    docs_text = "\n\n".join([f"Source ({d['metadata'].get('source', 'unknown')}): {d['page_content']}" for d in state["retrieved_docs"]])
    web_text = state.get("web_search_results", "")

    prompt = f"""You are an expert Research Assistant. Synthesize a comprehensive, rigorous, and accurate answer to the user's query based on the provided research paper excerpts and optional web search findings.

User Query: {query}

Retrieved Local Papers Excerpts:
{docs_text}

Web Search Results:
{web_text}

Provide a well-structured synthesis with citations to sources.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "draft_answer": response.content}

def fact_check_node(state: AgentState) -> AgentState:
    draft = state["draft_answer"]
    docs_text = "\n\n".join([d['page_content'] for d in state["retrieved_docs"]])

    prompt = f"""You are a rigorous Fact-Checking Agent. Review the draft answer against the retrieved source excerpts to ensure no hallucination or unsupported claims exist.
If there are unsupported claims, correct or caveat them. If everything is well-supported, confirm validity.

Draft Answer:
{draft}

Source Excerpts:
{docs_text}

Provide the verified/corrected answer:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "final_answer": response.content, "fact_check_result": "Passed"}

def follow_up_node(state: AgentState) -> AgentState:
    final_answer = state["final_answer"]

    prompt = f"""Based on the following research answer, generate 3 smart, probing follow-up research questions that the user might want to explore next. Return them as a numbered list.

Answer:
{final_answer}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    questions = [q.strip() for q in response.content.split("\n") if q.strip() and q[0].isdigit()]
    return {**state, "follow_ups": questions}

# Build LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("retrieval", retrieval_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("fact_check", fact_check_node)
workflow.add_node("follow_up", follow_up_node)

workflow.set_entry_point("retrieval")
workflow.add_edge("retrieval", "web_search")
workflow.add_edge("web_search", "synthesis")
workflow.add_edge("synthesis", "fact_check")
workflow.add_edge("fact_check", "follow_up")
workflow.add_edge("follow_up", END)

app_graph = workflow.compile()
