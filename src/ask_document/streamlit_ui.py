"""Streamlit user interface for Ask the Document."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .answer_chain import answer_question
from .ingestion import load_documents
from .prompts import REFUSAL_TEXT


DOCUMENTS_PATH = Path("data/raw/documents.csv")


@st.cache_data
def get_documents() -> pd.DataFrame:
    return load_documents(DOCUMENTS_PATH)


def _show_citations(result: dict) -> None:
    citations = result.get("citations", [])
    chunks = result.get("chunks", [])
    if not citations:
        return

    st.subheader("Sources")
    for citation_number, chunk in zip(citations, chunks):
        metadata = chunk.get("metadata", {})
        label = f"Chunk {citation_number} · {metadata.get('chunk_id', chunk.get('chunk_id', 'unknown'))}"
        with st.expander(label):
            st.write(chunk.get("text", ""))
            source_url = metadata.get("source_url")
            if source_url:
                st.caption(f"Source: {source_url}")


def run() -> None:
    st.set_page_config(page_title="Ask the Document", page_icon="📄", layout="wide")
    st.title("Ask the Document")
    st.write("Select one document and ask a question grounded only in that document.")

    try:
        documents = get_documents()
    except Exception as exc:
        st.error(f"Could not load documents: {exc}")
        return

    with st.sidebar:
        st.header("Query settings")
        selected_index = st.selectbox(
            "Document",
            options=documents["index"].tolist(),
            format_func=lambda value: f"Document {value}",
        )
        method = st.radio(
            "Retrieval method",
            options=["hybrid", "dense"],
            format_func=lambda value: "Hybrid · dense + BM25 + RRF" if value == "hybrid" else "Dense · ChromaDB",
        )

    question = st.text_area(
        "Question",
        placeholder="Ask a question about the selected document...",
        height=120,
    )

    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving evidence and generating an answer..."):
            try:
                result = answer_question(
                    question=question,
                    document_index=int(selected_index),
                    method=method,
                )
            except Exception as exc:
                st.error(f"The question could not be answered: {exc}")
                return

        if result["answer"] == REFUSAL_TEXT:
            st.warning(REFUSAL_TEXT)
        else:
            st.subheader("Answer")
            st.write(result["answer"])
        _show_citations(result)
