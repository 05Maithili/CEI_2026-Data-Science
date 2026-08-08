"""
ingestion.py
Handles loading raw document files (PDF / TXT) and web URLs.
Splits text into 500-character chunks with 100-character overlap using RecursiveCharacterTextSplitter.
Guarantees page 0 (Page 1) metadata preservation across all loaders and tags reference sections.
"""

import os
import re
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Pattern to identify reference and bibliography sections
REF_PATTERN = re.compile(
    r"^\s*(references|refernces|bibliography|works\s+cited)\b",
    re.IGNORECASE | re.MULTILINE
)


def load_uploaded_file(uploaded_file) -> List[Document]:
    """
    Saves an uploaded Streamlit file to a temporary location, loads document content
    page-by-page with metadata (source filename, page number), and cleans up temporary files.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    elif suffix == ".txt":
        loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["page"] = 0
    else:
        os.remove(tmp_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    # Ensure page metadata is cleanly normalized (0-indexed integer)
    for idx, d in enumerate(docs):
        d.metadata["source"] = uploaded_file.name
        if "page" not in d.metadata:
            if "page_number" in d.metadata:
                d.metadata["page"] = max(0, int(d.metadata["page_number"]) - 1)
            else:
                d.metadata["page"] = idx

    os.remove(tmp_path)
    return docs


def load_from_url(url: str) -> List[Document]:
    """
    Scrapes main text content from a web URL, returning Document objects tagged with source URL and page=0.
    """
    try:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = url
            d.metadata["page"] = 0
        return docs
    except ImportError as e:
        raise ImportError(
            "Web URL loading requires beautifulsoup4. Run: pip install beautifulsoup4 requests"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to load URL '{url}': {e}") from e


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Splits documents into overlapping chunks (chunk_size=500, chunk_overlap=100)
    while preserving page numbers, source metadata, and tagging reference sections.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks: List[Document] = []

    # Process page by page so page boundaries and page metadata are preserved
    for doc in documents:
        page_num = doc.metadata.get("page", 0)
        source_name = doc.metadata.get("source", "document")

        chunks = splitter.split_documents([doc])
        for c in chunks:
            c.metadata["source"] = source_name
            c.metadata["page"] = page_num

            # Detect and tag bibliography / reference chunks
            is_ref = bool(REF_PATTERN.search(c.page_content))
            c.metadata["is_reference"] = is_ref

            final_chunks.append(c)

    return final_chunks