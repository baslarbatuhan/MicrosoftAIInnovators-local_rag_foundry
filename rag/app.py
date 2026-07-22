"""
Streamlit web arayüzü — CLI ile aynı pipeline (answer_query + streaming callback).

Çalıştırma:  python -m streamlit run rag/app.py   (repo kökünden)

Konuşma geçmişi sadece görüntüleme amaçlıdır — modele geçmiş gönderilmez
(tek-tur Q&A, conversational memory kapsam dışı).
"""
import os
import sys

# Streamlit bu dosyayı bir SCRIPT olarak çalıştırır (sys.path[0] = rag/), paket olarak değil.
# `from rag...` import'larının çalışması için repo kökünü sys.path'e ekle.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from rag.core import answer_query, load_chat_client
from rag.retrieval import get_top_chunks

st.set_page_config(page_title="Local RAG Assistant", page_icon="📚")


@st.cache_resource(show_spinner="Model yükleniyor (ilk açılışta ~30 sn sürebilir)...")
def get_chat_client():
    """Model bir kez yüklenir, Streamlit yeniden çalıştırmalarında cache'ten gelir."""
    return load_chat_client()


chat_client = get_chat_client()

st.title("📚 Foundry Local Documentation Assistant")
st.caption(
    "Resmi Foundry Local dokümantasyonundan cevap verir — Foundry Local üzerinde, tamamen "
    "çevrimdışı çalışır, kaynak gösterir, bilmediğinde uydurmaz. Soruları İngilizce sor "
    '(ör. "How do I install Foundry Local?").'
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
            st.caption("Kaynaklar: " + ", ".join(entry["sources"]))

question = st.chat_input("Sorunu yaz (İngilizce)...")
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

        answer, sources = answer_query(question, chat_client, on_delta=on_delta)
        placeholder.markdown(answer)

        if sources:
            st.caption("Kaynaklar: " + ", ".join(sources))

        # Şeffaflık: modelin gördüğü chunk'lar (CLI'daki --verbose'un UI karşılığı)
        with st.expander("🔍 Retrieval detayı — modelin gördüğü chunk'lar"):
            for i, chunk in enumerate(get_top_chunks(question), start=1):
                st.markdown(f"**#{i} — {chunk['source']}** (benzerlik: {chunk['score']:.3f})")
                st.text(chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""))

    st.session_state.history.append(
        {"question": question, "answer": answer, "sources": sources}
    )
