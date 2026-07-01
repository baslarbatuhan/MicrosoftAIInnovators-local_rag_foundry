import json
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
DB_PATH = "rag_demo.db"

SENTENCES = [
    "Foundry Local, LLM'leri cihaz üzerinde çalıştırmayı sağlar.",
    "SQLite, sunucu gerektirmeyen hafif bir veritabanıdır.",
    "Kediler evcil hayvan olarak çok sevilir.",
    "RAG, retrieval ve generation adımlarını birleştirir.",
]


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL  -- JSON-serialized vector
        )
    """)
    conn.commit()


def insert_document(conn: sqlite3.Connection, content: str, embedding: list[float]) -> None:
    conn.execute(
        "INSERT INTO documents (content, embedding) VALUES (?, ?)",
        (content, json.dumps(embedding)),
    )
    conn.commit()


def get_all_documents(conn: sqlite3.Connection) -> list[tuple[int, str, list[float]]]:
    rows = conn.execute("SELECT id, content, embedding FROM documents").fetchall()
    return [(row_id, content, json.loads(embedding_json)) for row_id, content, embedding_json in rows]


def main():
    config = Configuration(app_name="LocalRagAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    model.load()
    embed_client = model.get_embedding_client()

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    # Tablo boşsa doldur (tekrar çalıştırılınca yeniden embed etmesin)
    row_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if row_count == 0:
        print("Veritabanı boş, cümleler embed edilip kaydediliyor...")
        response = embed_client.generate_embeddings(SENTENCES)
        for sentence, item in zip(SENTENCES, response.data):
            insert_document(conn, sentence, item.embedding)
        print(f"{len(SENTENCES)} kayıt eklendi.")
    else:
        print(f"Veritabanında zaten {row_count} kayıt var, embed atlandı.")

    print("\n--- Veritabanındaki kayıtlar ---")
    for row_id, content, embedding in get_all_documents(conn):
        print(f"id={row_id}  vektör_boyutu={len(embedding)}  içerik={content}")

    conn.close()


if __name__ == "__main__":
    main()
