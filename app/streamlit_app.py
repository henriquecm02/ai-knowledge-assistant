"""Assistente de conhecimento: converse com os documentos internos, com a fonte de cada resposta.

  streamlit run app/streamlit_app.py
Sem GEMINI_API_KEY, funciona em modo busca (BM25): mostra os trechos mais relevantes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from kassist.assistant import Assistant
from kassist.ingest import load_corpus
from kassist.retrieval import Index

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(Path.home() / ".config" / "portfolio" / ".env")
KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Assistente de conhecimento", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Indexando documentos…")
def build(folder: str, use_llm: bool):
    chunks = load_corpus(Path(folder))
    if use_llm:
        from kassist.llm import GeminiLLM
        from kassist.retrieval import GeminiEmbedder
        return Index(chunks, GeminiEmbedder(KEY)), GeminiLLM(KEY, os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    return Index(chunks), None


with st.sidebar:
    st.header("📚 Base de conhecimento")
    uploads = st.file_uploader("Usar meus documentos (PDF, DOCX, MD, TXT)", type=["pdf", "docx", "md", "txt"],
                               accept_multiple_files=True)
    folder = ROOT / "corpus"
    if uploads:
        tmp = Path(tempfile.mkdtemp(prefix="kassist_"))
        for f in uploads:
            (tmp / Path(f.name).name).write_bytes(f.getvalue())
        folder = tmp
    use_llm = bool(KEY) and st.toggle("Usar IA (Gemini)", value=bool(KEY), disabled=not KEY,
                                      help=None if KEY else "Defina GEMINI_API_KEY para respostas geradas.")
    rerank = use_llm and st.toggle("Reordenar trechos com IA", value=False)
    index, llm = build(str(folder), use_llm)
    docs = sorted({c.doc for c in index.chunks})
    st.caption(f"{len(docs)} documentos · {len(index.chunks)} trechos · busca "
               f"{'híbrida (BM25 + embeddings)' if index.vectors is not None else 'BM25'}")
    for d in docs:
        st.markdown(f"- `{d}`")

st.title("Assistente de conhecimento")
st.caption("Respostas baseadas só nos documentos, sempre com a fonte. "
           + ("" if use_llm else "**Modo busca:** sem IA, são mostrados os trechos mais relevantes."))

assistant = Assistant(index, llm, k=4, rerank=rerank)
if "hist" not in st.session_state:
    st.session_state.hist = []

exemplos = ["Qual o limite de hospedagem em capitais?", "Ganho folga no mês do meu aniversário?",
            "Meu notebook foi roubado, o que eu faço?"]
cols = st.columns(len(exemplos))
clicked = next((e for c, e in zip(cols, exemplos) if c.button(e, use_container_width=True)), None)
question = st.chat_input("Pergunte sobre as políticas da empresa") or clicked

for item in st.session_state.hist:
    with st.chat_message("user"):
        st.markdown(item["pergunta"])
    with st.chat_message("assistant"):
        st.markdown(item["resposta"])

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        ans = assistant.ask(question)
        st.markdown(ans.resposta)
        for w in ans.avisos:
            if "sem LLM" not in w:
                st.warning(w)
        for s in ans.fontes:
            with st.expander(f"[{s['n']}] {s['citacao']}", expanded=not use_llm and s["n"] == 1):
                st.markdown(s["texto"])
        with st.expander("Como a resposta foi encontrada"):
            st.dataframe(ans.trechos_recuperados, use_container_width=True, hide_index=True)
    st.session_state.hist.append({"pergunta": question, "resposta": ans.resposta})
