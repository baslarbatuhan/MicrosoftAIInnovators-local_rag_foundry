import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

from retrieval import get_top_chunks

CHAT_MODEL_ALIAS = "phi-3.5-mini"

# Bilgi tabanı ve sorular İngilizce olduğu için system prompt ve "bilmiyorum"
# cümlesi de İngilizce — dil tutarlılığı modelin davranışını çok iyileştiriyor.
# Cümleyi sabit tutmak "bilmiyorum" durumunu güvenilir tespit etmemizi sağlıyor.
NO_ANSWER = "I don't have that information."

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the information in the "
    "CONTEXT below. Do not use outside knowledge and do not guess. Keep your answer concise. "
    f"If the answer is not in the context, reply with exactly this sentence and nothing else: "
    f"\"{NO_ANSWER}\""
)


def build_context(chunks: list[dict]) -> str:
    parts = [f"[Source: {chunk['source']}]\n{chunk['content']}" for chunk in chunks]
    return "\n\n---\n\n".join(parts)


def is_refusal(answer: str) -> bool:
    """Model 'bilmiyorum' dedi mi?

    Hafta 5 turlarında öğrenilen üç ders:
    - Sondaki noktayı sabit arama kaçırıyor (model cümleye devam edebiliyor).
    - Model NO_ANSWER'ı bazen apostrofsuz yazıyor: "I do not have that information".
    - Model bazen kendi kelimeleriyle reddediyor ("The context does not provide...").
      Bu kalıplar sadece cevabın BAŞINDA aranmalı — geçerli cevaplar bazen sonda
      "...the exact comparison is not provided" gibi uyarılarla bitiyor, onları
      reddetme saymak yanlış olur.
    """
    text = answer.lower()
    if "i don't have that information" in text or "i do not have that information" in text:
        return True

    head = text[:120]
    head_markers = (
        "the context does not provide",
        "the context provided does not",
        "the provided context does not",
    )
    return any(marker in head for marker in head_markers)


def answer_query(question: str, chat_client, top_k: int = 3, verbose: bool = False) -> tuple[str, list[str]]:
    """Retrieval (get_top_chunks) + generation (chat_client) adımlarını birleştirir.

    (cevap_metni, kaynak_listesi) döner. Kaynaklar modelden değil, retrieval'ın
    gerçekten getirdiği chunk'lardan deterministik olarak çıkarılır. Model
    'bilmiyorum' derse kaynak listesi boş döner (yanıltıcı atıf olmasın diye).
    verbose=True ise retrieval'ın getirdiği chunk'ları loglar (PDF Hafta 4:
    "log retrieved chunks for verification").
    """
    # Edge case: boş/boşluk soru — embedding client boş string'e ValueError
    # fırlatıyor, LLM'e gitmeden kibarca reddet.
    question = question.strip()
    if not question:
        return NO_ANSWER, []

    chunks = get_top_chunks(question, top_k=top_k)
    if verbose:
        print(f"\n[retrieval] top_{top_k} chunk:")
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk["content"][:80].replace("\n", " ")
            print(f"  #{i} score={chunk['score']:.4f} source={chunk['source']}  {preview}...")

    context = build_context(chunks)

    response = chat_client.complete_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ])
    answer = response.choices[0].message.content.strip()

    if is_refusal(answer):
        return answer, []

    # Benzersiz kaynaklar, retrieval sırasını koruyarak (dict.fromkeys sırayı korur)
    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))
    return answer, sources


def load_chat_client():
    config = Configuration(app_name="LocalRagAssistant")
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model bulunamadı: {CHAT_MODEL_ALIAS}")

    if not model.is_cached:
        print(f"'{CHAT_MODEL_ALIAS}' indiriliyor...")
        model.download(progress_callback=lambda pct: print(f"  %{pct:.1f}", end="\r"))
        print()

    model.load()
    chat_client = model.get_chat_client()
    # Yanıtı sınırla: hem gecikmeyi düşürür hem modelin gereksiz uzun/dağınık
    # cevaplar üretmesini engeller (Q&A için kısa cevap zaten daha iyi).
    chat_client.settings.max_tokens = 256
    # Deterministik çıktı: aynı soru her seferinde aynı cevabı versin. Değerlendirmenin
    # tekrarlanabilir olması ve sınırdaki soruların turdan tura zıplamaması için.
    chat_client.settings.temperature = 0.0
    return chat_client


def main():
    # --verbose bayrağı: retrieval'ın getirdiği chunk'ları göster (demo/hata ayıklama)
    verbose = "--verbose" in sys.argv

    print("Model yükleniyor...")
    chat_client = load_chat_client()

    print("Local RAG Assistant hazır. Çıkmak için 'exit' yaz.\n")
    while True:
        question = input("Soru: ").strip()
        if question.lower() in ("exit", "quit", "çık"):
            break
        if not question:
            continue

        answer, sources = answer_query(question, chat_client, verbose=verbose)
        print(f"\nCevap: {answer}")
        if sources:
            print(f"Kaynaklar: {', '.join(sources)}")
        print()


if __name__ == "__main__":
    main()
