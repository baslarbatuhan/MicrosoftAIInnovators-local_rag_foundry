"""
Ingestion pipeline: documents/ klasöründeki .txt dosyalarını okur,
paragraf bazlı chunk'lara böler, her chunk için embedding üretir
ve SQLite veritabanına (rag.db) yazar.

Yeniden çalıştırmak güvenlidir: tabloyu temizleyip documents/ klasörünün
güncel haline göre yeniden doldurur.
"""
import glob
import json
import os
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
DOCUMENTS_DIR = "documents"
DB_PATH = "data/rag.db"
MAX_CHUNK_CHARS = 800


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL  -- JSON-serialized vector
        )
    """)
    conn.commit()


def read_document(path: str) -> tuple[str, str]:
    """Dosyayı okur. İlk satır 'Kaynak: <url>' ise onu ayırır, geri kalan gövdeyi döner."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n", 2)
    if lines[0].startswith("Kaynak:"):
        source_url = lines[0].removeprefix("Kaynak:").strip()
        body = lines[2] if len(lines) > 2 else ""
    else:
        source_url = os.path.basename(path)
        body = text

    return source_url, body


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Metni boş satırlara göre paragraflara ayırır, ardışık paragrafları
    max_chars'ı aşmayacak şekilde birleştirerek chunk'lar oluşturur.

    Overlap: yeni chunk, önceki chunk'ın SON paragrafıyla başlar. Hafta 5
    değerlendirmesinde başlık/tarih satırlarının ("Required features:",
    "Wednesday, April 3 - v0.2.7 Update") ait oldukları içerikten ayrı
    chunk'lara düştüğü görüldü — overlap bu sınır kopmalarını yumuşatıyor.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []

    def current_len(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + 2 * max(len(parts) - 1, 0)

    for para in paragraphs:
        if current and current_len(current) + 2 + len(para) > max_chars:
            chunks.append("\n\n".join(current))
            # Overlap: önceki chunk'ın son paragrafını yeni chunk'a taşı
            # (sığıyorsa — tek başına max_chars'ı aşan dev paragrafları taşıma)
            last = current[-1]
            current = [last, para] if len(last) + 2 + len(para) <= max_chars else [para]
        else:
            current.append(para)
    if current:
        chunks.append("\n\n".join(current))

    return chunks


def main():
    config = Configuration(app_name="LocalRagAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model bulunamadı: {EMBEDDING_MODEL_ALIAS}")
    model.load()
    embed_client = model.get_embedding_client()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    conn.execute("DELETE FROM documents")

    total_chunks = 0
    for path in sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt"))):
        filename = os.path.basename(path)
        source_url, body = read_document(path)
        chunks = chunk_text(body)

        print(f"{filename}: {len(chunks)} chunk")
        response = embed_client.generate_embeddings(chunks)

        for chunk, item in zip(chunks, response.data):
            conn.execute(
                "INSERT INTO documents (source, content, embedding) VALUES (?, ?, ?)",
                (filename, chunk, json.dumps(item.embedding)),
            )
        total_chunks += len(chunks)

    conn.commit()

    row_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"\nToplam {total_chunks} chunk üretildi, veritabanında {row_count} satır var.")
    assert total_chunks == row_count, "Chunk sayısı ile DB satır sayısı uyuşmuyor!"

    conn.close()


if __name__ == "__main__":
    main()
