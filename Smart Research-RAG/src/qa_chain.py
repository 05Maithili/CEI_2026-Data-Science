"""
qa_chain.py
Conversational RAG chain with strict grounded prompting, chunk logging,
and automatic fallback from Google Gemini to local Ollama on API/Quota errors.
"""

import logging
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Configure logger for retrieved chunk inspection
logger = logging.getLogger("RAGChain")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# ── Strict Grounded System Prompt ──────────────────────────────────────────────

STRICT_QA_SYSTEM = (
    "You are a research assistant.\n\n"
    "Answer ONLY from the retrieved document context.\n\n"
    "Rules:\n"
    "* Do not use prior knowledge.\n"
    "* Do not infer from references or bibliography.\n"
    "* Extract factual information exactly from the document.\n"
    "* If the answer is not present in the retrieved context, say:\n"
    "  'The answer is not available in the retrieved document.'\n"
    "* Keep answers concise and accurate.\n"
    "* Always include source filename and page number.\n\n"
    "Retrieved Document Context:\n{context}"
)

REFORMULATE_SYSTEM = (
    "Given the conversation history and the latest user question, "
    "formulate a concise standalone search query that captures the user's intent. "
    "Return ONLY the reformulated query — no explanation, no preamble."
)


# ── LLM Factory ───────────────────────────────────────────────────────────────

def _get_llm(provider: str, model_name: str, temperature: float = 0.0):
    """Instantiates LLM model for Google Gemini or local Ollama."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # fallback # type: ignore
        return ChatOllama(model=model_name, temperature=temperature)

    raise ValueError(f"Unknown provider: '{provider}'. Choose 'gemini' or 'ollama'.")


def _format_docs(docs) -> str:
    """Formats list of Document objects into a clean text block for the prompt."""
    formatted_blocks = []
    for d in docs:
        src = d.metadata.get("source", "document")
        pg = d.metadata.get("page", 0) + 1
        formatted_blocks.append(f"[Source: {src} | Page: {pg}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted_blocks)


class RAGChainWrapper:
    """
    RAG Execution Chain supporting MMR retrieval, logging, and automatic
    Gemini -> Ollama fallback on quota limits (429 RESOURCE_EXHAUSTED).
    """

    def __init__(self, retriever, model_name: str = "gemini-3.1-flash-lite", temperature: float = 0.0, provider: str = "gemini"):
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider
        self.primary_llm = _get_llm(provider, model_name, temperature)

        # Reformulation and Q&A Prompts
        self.reformulate_prompt = ChatPromptTemplate.from_messages([
            ("system", REFORMULATE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", STRICT_QA_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ])

    def invoke(self, inputs: dict) -> dict:
        question = inputs["question"]
        chat_history = inputs.get("chat_history") or []

        # Step 1: Reformulate query if chat history exists
        if chat_history:
            try:
                search_query = (self.reformulate_prompt | self.primary_llm | StrOutputParser()).invoke(
                    {"question": question, "chat_history": chat_history}
                )
            except Exception:
                search_query = question
        else:
            search_query = question

        # Step 2: Retrieve relevant chunks
        docs = self.retriever.invoke(search_query)

        # Logging: Inspect retrieved chunks before LLM generation
        logger.info(f"Query: '{question}' (Search Query: '{search_query}')")
        logger.info(f"Retrieved {len(docs)} chunks:")
        for idx, d in enumerate(docs, 1):
            src = d.metadata.get("source", "document")
            pg = d.metadata.get("page", 0) + 1
            score = d.metadata.get("similarity_score", 0.0)
            snippet = d.page_content[:100].replace("\n", " ")
            logger.info(f"  Chunk {idx}: [{src} p.{pg} | score={score}] '{snippet}...'")

        formatted_context = _format_docs(docs)
        prompt_val = self.qa_prompt.invoke({
            "context": formatted_context,
            "chat_history": chat_history,
            "question": question,
        })

        # Step 3: Generate answer with automatic Gemini -> Ollama fallback
        try:
            answer = (self.primary_llm | StrOutputParser()).invoke(prompt_val)
        except Exception as exc:
            err_msg = str(exc)
            if self.provider == "gemini" and ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower()):
                logger.warning("Gemini API quota exhausted. Falling back to local Ollama (llama3.2)...")
                fallback_llm = _get_llm("ollama", "llama3.2", self.temperature)
                answer = (fallback_llm | StrOutputParser()).invoke(prompt_val)
            else:
                raise exc

        return {
            "answer": answer,
            "sources": docs,
        }


def build_qa_chain(
    retriever,
    model_name: str = "gemini-3.1-flash-lite",
    temperature: float = 0.0,
    provider: str = "gemini",
):
    """Constructs the RAGChainWrapper instance."""
    return RAGChainWrapper(retriever, model_name=model_name, temperature=temperature, provider=provider)


def ask_question(chain, query: str, chat_history=None) -> Dict[str, Any]:
    """
    Executes the RAG chain and returns structured dictionary:
      - answer: str
      - sources: List[Document]
      - page_numbers: List[int]
      - retrieved_context: List[Document]
    """
    result = chain.invoke({
        "question": query,
        "chat_history": chat_history or [],
    })

    docs = result["sources"]
    page_numbers = []
    for d in docs:
        pg = d.metadata.get("page", 0) + 1
        if pg not in page_numbers:
            page_numbers.append(pg)

    return {
        "answer": result["answer"],
        "sources": docs,
        "page_numbers": page_numbers,
        "retrieved_context": docs,
    }