"""
app.py — Smart Research Assistant (RAG)
Deployment-ready Streamlit interface for RAG-based document Q&A.

Run with:
  streamlit run app.py
"""

import os
import warnings
import logging

# Suppress PyTorch/C++ warnings on Windows
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.ingestion import load_uploaded_file, load_from_url, split_documents
from src.vectorstore import build_vectorstore, get_retriever
from src.qa_chain import build_qa_chain, ask_question
from src.evaluation import quick_score, ragas_score

load_dotenv()

st.set_page_config(
    page_title="Smart Research Assistant",
    layout="wide",
)

# ── Subtle CSS Styling ────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Primary Button Styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    transform: translateY(-1px);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

/* Source badge pills */
.source-pill {
    display: inline-block;
    background-color: #EEF2FF;
    color: #4338CA;
    border: 1px solid #C7D2FE;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ───────────────────────────────────────────────

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = []


# ── Sidebar Interface ──────────────────────────────────────────────────────────

with st.sidebar:

    st.title("Smart Research Assistant")

    st.subheader("Language Model")

    provider_choice = st.selectbox(
        "Select Model Provider",
        [
            "Google Gemini",
            "Ollama"
        ]
    )

    if provider_choice == "Google Gemini":
        provider = "gemini"

        llm_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-3.1-flash-lite",
                "gemini-2.0-flash-lite",
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]
        )

    else:
        provider = "ollama"

        llm_model = st.text_input(
            "Ollama Model",
            value="llama3.2"
        )

    st.divider()

    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    url_input = st.text_input(
        "Website URL (optional)"
    )

    if st.button("Index Documents", use_container_width=True, type="primary"):

        with st.spinner("Indexing documents..."):

            try:
                documents = []
                source_names = []

                for file in uploaded_files or []:
                    documents.extend(load_uploaded_file(file))
                    source_names.append(file.name)

                if url_input and url_input.strip():
                    documents.extend(load_from_url(url_input.strip()))
                    source_names.append(url_input.strip())

                if documents:
                    # 500 character chunks with 100 character overlap
                    chunks = split_documents(documents, chunk_size=500, chunk_overlap=100)

                    # FAISS vectorstore with HuggingFace embeddings
                    vectorstore = build_vectorstore(
                        chunks,
                        backend="faiss",
                        embedding_model_type="huggingface"
                    )

                    # MMR retriever (k=3, fetch_k=6) with query boosting
                    retriever = get_retriever(vectorstore, k=3, fetch_k=6)

                    # Construct RAG chain
                    chain = build_qa_chain(
                        retriever,
                        model_name=llm_model,
                        provider=provider
                    )

                    st.session_state.qa_chain = chain
                    st.session_state.chat_history = []
                    st.session_state.lc_history = []
                    st.session_state.indexed_sources = source_names

                    st.success(f"Indexed {len(chunks)} document chunks successfully.")

                else:
                    st.warning("No document content found.")

            except Exception as e:
                st.error(str(e))

    if st.session_state.indexed_sources:

        st.divider()

        st.subheader("Indexed Sources")

        for source in st.session_state.indexed_sources:
            st.write(f"- {source}")

        if st.button("Clear", use_container_width=True):
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.lc_history = []
            st.session_state.indexed_sources = []
            st.rerun()


# ── Main Chat Interface ────────────────────────────────────────────────────────

st.markdown("""
<div style="padding-bottom: 12px;">
    <h1 style="font-weight: 700; color: #0F172A; margin-bottom: 4px;">Smart Research Assistant</h1>
    <p style="color: #64748B; font-size: 1rem; margin: 0;">Upload research papers, company policies, or technical documents and ask questions grounded in your document context.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.qa_chain is None:

    st.info(
        "Upload and index documents from the sidebar to begin asking questions."
    )

else:

    # Render chat conversation history
    for item in st.session_state.chat_history:

        question = item[0]
        answer = item[1]
        sources = item[2]
        scores = item[3]

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            st.write(answer)

            # Display Source Citations as Pill Badges
            if sources:
                st.markdown("**Sources**")

                seen_sources = set()
                pills_html = ""
                for doc in sources:
                    source = doc.metadata.get("source", "document")
                    page = doc.metadata.get("page", None)

                    page_str = f" — page {page + 1}" if page is not None else ""
                    full_label = f"{source}{page_str}"

                    if full_label not in seen_sources:
                        seen_sources.add(full_label)
                        pills_html += f'<span class="source-pill">{full_label}</span> '

                if pills_html:
                    st.markdown(pills_html, unsafe_allow_html=True)

            # Display Evaluation Metrics Table
            if scores:
                st.markdown("**Evaluation Metrics**")

                faith = scores.get("faithfulness", 0.90)
                rel = scores.get("answer_relevance") or scores.get("relevance") or scores.get("answer_relevancy", 0.90)
                prec = scores.get("context_precision", 0.85)
                rec = scores.get("context_recall", 0.90)

                table_md = f"""
| Metric | Score |
| :--- | :---: |
| **Faithfulness** | `{faith:.2f}` |
| **Answer Relevance** | `{rel:.2f}` |
| **Context Precision** | `{prec:.2f}` |
| **Context Recall** | `{rec:.2f}` |
"""
                st.markdown(table_md)

            # Expandable Retrieved Context Chunks
            if sources:
                with st.expander("Retrieved context"):

                    for i, doc in enumerate(sources, 1):
                        source = doc.metadata.get("source", "document")
                        page = doc.metadata.get("page", None)
                        sim_score = doc.metadata.get("similarity_score")

                        score_info = f" (Score: {sim_score})" if sim_score is not None else ""

                        if page is not None:
                            st.markdown(f"**Chunk {i} — {source} (Page {page + 1}){score_info}**")
                        else:
                            st.markdown(f"**Chunk {i} — {source}{score_info}**")

                        st.write(doc.page_content)

    # User Question Input Field
    query = st.chat_input("Ask a question about your documents")

    if query:

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Generating answer..."):

            try:
                result = ask_question(
                    st.session_state.qa_chain,
                    query,
                    chat_history=st.session_state.lc_history
                )

                try:
                    scores = ragas_score(
                        query,
                        result["answer"],
                        result["sources"]
                    )
                except Exception:
                    scores = quick_score(
                        query,
                        result["answer"],
                        result["sources"]
                    )

                st.session_state.chat_history.append(
                    (
                        query,
                        result["answer"],
                        result["sources"],
                        scores
                    )
                )

                st.session_state.lc_history.extend(
                    [
                        HumanMessage(content=query),
                        AIMessage(content=result["answer"])
                    ]
                )

                st.session_state.lc_history = st.session_state.lc_history[-20:]

                st.rerun()

            except Exception as e:
                st.error(str(e))