import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Assistant with RAG & Multi-Agent Orchestration")
st.markdown("Upload research papers (PDFs) and interact with a multi-agent system featuring **Retrieval**, **Synthesis**, **Fact-Checking**, **Web Search Fallback**, and **Follow-up Generation**.")

# Sidebar for PDF uploads
st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Ingest PDF"):
        with st.spinner("Processing and vectorizing PDF..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{BACKEND_URL}/upload", files=files)
                if res.status_code == 200:
                    data = res.json()
                    st.sidebar.success(f"Successfully ingested {data['filename']} ({data['chunks_ingested']} chunks).")
                else:
                    st.sidebar.error(f"Error: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.sidebar.error(f"Could not connect to backend: {e}")

# Main Chat Interface
st.header("💬 Research Conversation")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your research papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Multi-Agent pipeline at work (Retrieval → Web Search → Synthesis → Fact-Check → Follow-ups)..."):
            try:
                res = requests.post(f"{BACKEND_URL}/query", json={"query": prompt})
                if res.status_code == 200:
                    data = res.json()
                    answer = data["final_answer"]
                    st.markdown(answer)

                    # Expandable details for debugging / transparency
                    with st.expander("🔍 Multi-Agent Execution Details"):
                        st.subheader("Retrieved Context Chunks")
                        for i, doc in enumerate(data["retrieved_docs"]):
                            st.markdown(f"**Source:** {doc['metadata'].get('source')} | **Score:** {doc['score']:.4f}")
                            st.text(doc["page_content"][:300] + "...")

                        st.subheader("Web Search Fallback Results")
                        st.text(data["web_search_results"])

                        st.subheader("Suggested Follow-up Questions")
                        for q in data["follow_ups"]:
                            st.markdown(f"- {q}")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    err_msg = res.json().get("detail", "Unknown error")
                    st.error(f"Error: {err_msg}")
            except Exception as e:
                    st.error(f"Failed to communicate with backend: {e}")
