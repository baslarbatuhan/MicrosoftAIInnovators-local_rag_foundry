"""
Retrieval: kullanıcı sorgusunu embed eder, rag.db'deki tüm chunk embeddinglerine
karşı cosine similarity hesaplar ve en alakalı top_k chunk'ı döner.
"""
import json
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
DB_PATH = "data/rag.db"


def get_manager() -> FoundryLocalManager:
    """FoundryLocalManager zaten başlatılmışsa (ör. main.py tarafından) mevcut
    instance'ı kullanır, değilse burada başlatır. Singleton'ı iki kez
    initialize etmeye çalışmak hataya neden olduğu için bu kontrol gerekli."""
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="LocalRagAssistant"))
    return FoundryLocalManager.instance


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def get_top_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Verilen sorgu için veritabanındaki en alakalı top_k chunk'ı döner.
    Her sonuç {'source', 'content', 'score'} anahtarlarını içerir."""
    manager = get_manager()
    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    model.load()
    embed_client = model.get_embedding_client()

    query_embedding = embed_client.generate_embedding(query).data[0].embedding

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT source, content, embedding FROM documents").fetchall()
    conn.close()

    scored = [
        {"source": source, "content": content, "score": cosine_similarity(query_embedding, json.loads(embedding_json))}
        for source, content, embedding_json in rows
    ]
    scored.sort(key=lambda r: r["score"], reverse=True)

    return scored[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What do keybullet kin drop?",
        "What kind of gun does the bandana bullet kin use?",
        "What is llmware used for?",
    ]

    for query in test_queries:
        print(f"\nSORGU: {query}")
        for rank, chunk in enumerate(get_top_chunks(query, top_k=2), start=1):
            print(f"  #{rank}  score={chunk['score']:.4f}  source={chunk['source']}")
            print(f"       {chunk['content'][:150]}...")
