"""
Streamlit arayüzü — CLI ile aynı pipeline (answer_query + streaming).

Çalıştırma:  python -m streamlit run rag/app.py   (repo kökünden)

Geçmiş st.session_state.history'de tutuluyor; hem ekranda gösteriliyor hem de takip sorusunu
bağımsızlaştırmak (condensation) için answer_query'ye veriliyor. Cevap her zaman tek-tur üretilir.
"""
import os
import sys

# Streamlit bu dosyayı bir SCRIPT olarak çalıştırır (sys.path[0] = rag/), paket olarak değil.
# `from rag...` import'larının çalışması için repo kökünü sys.path'e ekle.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from rag.core import answer_query, load_chat_client

st.set_page_config(page_title="Local RAG Assistant", page_icon="📚")


@st.cache_resource(show_spinner="Loading model (first launch may take ~30 s)...")
def get_chat_client():
    """Model bir kez yüklenir, Streamlit yeniden çalıştırmalarında cache'ten gelir."""
    return load_chat_client()


chat_client = get_chat_client()

st.title("📚 Foundry Local Documentation Assistant")
st.caption(
    "Answers from the official Foundry Local documentation — running on Foundry Local, fully "
    "offline, with source attribution, and refusing to guess when the docs don't have the answer "
    '(e.g. "Does Foundry Local integrate with Slack?").'
)

if "history" not in st.session_state:
    st.session_state.history = []

# Önceki soru-cevapları göster (sayfa her etkileşimde yeniden çizilir)
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            st.caption("Sources: " + ", ".join(entry["sources"]))

question = st.chat_input("Ask a question (in English)...")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        # Streaming: token'lar geldikçe placeholder güncellenir
        placeholder = st.empty()
        parts: list[str] = []

        def on_delta(text: str) -> None:
            parts.append(text)
            placeholder.markdown("".join(parts) + "▌")

        # Şeffaflık paneli için modele giden gerçek sorgu ve chunk'ları yakala. Takip sorusu
        # condense edildiyse burada ham soru değil condense edilmiş hali görünüyor.
        retrieval: dict = {}

        def on_retrieval(search_query: str, chunks: list) -> None:
            retrieval["query"] = search_query
            retrieval["chunks"] = chunks

        # Geçmişi veriyoruz; takip sorusunu bağımsızlaştırmak için kullanılıyor (bu tur henüz eklenmedi)
        answer, sources = answer_query(
            question, chat_client, history=st.session_state.history,
            on_delta=on_delta, on_retrieval=on_retrieval,
        )
        placeholder.markdown(answer)

        if sources:
            st.caption("Sources: " + ", ".join(sources))

        # Modelin gördüğü chunk'lar (CLI'daki --verbose'un UI karşılığı). answer_query'nin kullandığı
        # sorgu ve chunk'lar gösterilir, ikinci bir retrieval yapılmaz; takip sorusu condense edildiyse
        # hangi sorguyla arandığı da burada görünür.
        with st.expander("🔍 Retrieval detail — the chunks the model saw"):
            search_query = retrieval.get("query")
            if search_query and search_query != question:
                st.caption(f"Follow-up rewritten for search: “{search_query}”")
            for i, chunk in enumerate(retrieval.get("chunks", []), start=1):
                st.markdown(f"**#{i} — {chunk['source']}** (similarity: {chunk['score']:.3f})")
                st.text(chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""))

    st.session_state.history.append(
        {"question": question, "answer": answer, "sources": sources}
    )
