from foundry_local_sdk import Configuration, FoundryLocalManager

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

SENTENCES = [
    "Foundry Local, LLM'leri cihaz üzerinde çalıştırmayı sağlar.",
    "SQLite, sunucu gerektirmeyen hafif bir veritabanıdır.",
    "Kediler evcil hayvan olarak çok sevilir.",
    "RAG, retrieval ve generation adımlarını birleştirir.",
]

QUERY = "Yerel bir dil modelini nasıl çalıştırırım?"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def find_relevant(query_embedding: list[float], sentence_embeddings: list[list[float]]) -> list[tuple[int, float]]:
    """Her cümle için benzerlik skorunu hesaplar, en yüksekten en düşüğe sıralar."""
    scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(sentence_embeddings)]
    return sorted(scores, key=lambda pair: pair[1], reverse=True)


def main():
    config = Configuration(app_name="LocalRagAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model bulunamadı: {EMBEDDING_MODEL_ALIAS}")

    if not model.is_cached:
        print(f"'{EMBEDDING_MODEL_ALIAS}' indiriliyor...")
        model.download(progress_callback=lambda pct: print(f"  %{pct:.1f}", end="\r"))
        print()

    print("Model yükleniyor...")
    model.load()

    embed_client = model.get_embedding_client()

    print("Örnek cümleler embed ediliyor...")
    sentence_response = embed_client.generate_embeddings(SENTENCES)
    sentence_embeddings = [item.embedding for item in sentence_response.data]

    print(f"Sorgu embed ediliyor: '{QUERY}'")
    query_response = embed_client.generate_embedding(QUERY)
    query_embedding = query_response.data[0].embedding

    ranked = find_relevant(query_embedding, sentence_embeddings)

    print("\nBenzerlik sıralaması (en alakalıdan en alakasıza):")
    for idx, score in ranked:
        print(f"  {score:.4f}  →  {SENTENCES[idx]}")


if __name__ == "__main__":
    main()
