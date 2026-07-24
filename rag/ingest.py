"""
Ingestion pipeline: doküman klasöründeki .txt dosyalarını okur, markdown başlıklarına göre
chunk'lara böler, her chunk için embedding üretir ve SQLite'a yazar.

Yeniden çalıştırmak güvenli: tabloyu sıfırlayıp klasörün güncel haline göre baştan dolduruyor.
"""
import glob
import json
import os
import re
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

from rag.config import (
    DB_PATH,
    DOCS_DIR as DOCUMENTS_DIR,
    EMBEDDING_MODEL_ALIAS,
    PROFILE,
)

MAX_CHUNK_CHARS = 800
# 0 = ardışık küçük bölümleri birleştirme, her bölüm kendi chunk'ı. Birleştirmeyi (min-merge)
# denedik ama tek satırlık kod bloğu gibi kısa ama önemli chunk'ları çevre metinle karıştırıp
# retrieval'ı zayıflattı. Gürültüyü boyuta göre değil hedefli (DATE_LINE_RE) temizliyoruz.
MIN_CHUNK_CHARS = 0

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Markdown link/tab sözdizimini sadeleştir: "[Windows](#tab/windows)" -> "Windows"
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Dataset hazırlama scripti her dokümanın başına "(Doküman tarihi: ...)" satırı ekliyor. Bu saf
# metadata; chunk'a girerse içeriksiz olmasına rağmen retrieval'da öne çıkıp modeli yanıltıyordu.
# Chunking sırasında atlıyoruz.
DATE_LINE_RE = re.compile(r"^\(Doküman tarihi:.*\)$")


def create_schema(conn: sqlite3.Connection) -> None:
    # Tabloyu her ingest'te sıfırdan kuruyoruz; ingest zaten baştan dolduruyor. Böylece şema
    # değişiklikleri (yeni kolon vb.) sorunsuz uygulanıyor.
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            heading_path TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            embedding TEXT NOT NULL  -- JSON-serialized vector
        )
    """)
    # BM25 keyword indeksi — hybrid retrieval adayları buradan geliyor. porter tokenizer kelime
    # köklerini eşliyor ("engine"/"engines", "list"/"lists").
    conn.execute("DROP TABLE IF EXISTS documents_fts")
    conn.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(content, tokenize='porter')")
    conn.commit()


def read_document(path: str) -> tuple[str, str]:
    """Dosyayı okur. İlk satır 'Kaynak: <url>' ise onu ayırır, geri kalan gövdeyi döner."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    lines = text.split("\n", 2)
    # İlk satır kaynak URL'siyse ayır. Mevcut korpus "Kaynak:" kullanıyor, dışarıdan eklenen
    # dosyalar "Source:" kullanabilir; ikisini de kabul ediyoruz.
    for prefix in ("Source:", "Kaynak:"):
        if lines[0].startswith(prefix):
            source_url = lines[0].removeprefix(prefix).strip()
            body = lines[2] if len(lines) > 2 else ""
            break
    else:
        source_url = os.path.basename(path)
        body = text

    return source_url, body


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Metni paragraflara ayırıp ardışık paragrafları max_chars'ı aşmadan birleştirir.

    Overlap: yeni chunk önceki chunk'ın son paragrafıyla başlar. Böylece bir başlık ya da giriş
    satırı içeriğinden ayrı bir chunk'a düştüğünde bağı kopmuyor.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []

    def current_len(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + 2 * max(len(parts) - 1, 0)

    for para in paragraphs:
        if current and current_len(current) + 2 + len(para) > max_chars:
            chunks.append("\n\n".join(current))
            # Önceki chunk'ın son paragrafını yeni chunk'a taşı (sığıyorsa; tek başına
            # max_chars'ı aşan dev paragrafları taşıma)
            last = current[-1]
            current = [last, para] if len(last) + 2 + len(para) <= max_chars else [para]
        else:
            current.append(para)
    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_markdown(body: str, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """Markdown başlık hiyerarşisine göre böler.

    Her chunk {'heading_path', 'content'} döner:
    - heading_path: "Doküman > Bölüm > Alt bölüm". Embedding'e eklenince kullanıcının doğal
      sorusuyla dokümanın teknik dili arasındaki boşluğu kapatıyor.
    - content: bölümün ham gövdesi. Uzun bölümler paragraf bazlı alt-chunk'lara bölünür,
      heading_path her birinde korunur.
    """
    heading_stack: dict[int, str] = {}
    sections: list[tuple[str, str]] = []
    buf: list[str] = []

    def current_path() -> str:
        return " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack))

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((current_path(), text))

    for line in body.split("\n"):
        match = HEADING_RE.match(line)
        if match:
            flush()
            buf.clear()
            level = len(match.group(1))
            heading_stack[level] = LINK_RE.sub(r"\1", match.group(2)).strip()
            for deeper in [lvl for lvl in heading_stack if lvl > level]:
                del heading_stack[deeper]
        elif DATE_LINE_RE.match(line.strip()):
            continue  # tarih metadata satırı — chunk'a girmesin (gürültü)
        else:
            buf.append(line)
    flush()

    # Ardışık küçük bölümleri MIN_CHUNK_CHARS'a ulaşana kadar birleştir (max'ı aşmadan).
    # Birleşen bölümlerin ortak başlık yolu chunk'ın path'i olur.
    chunks: list[dict] = []
    buf_paths: list[str] = []
    buf_texts: list[str] = []
    buf_len = 0

    def emit_buffer() -> None:
        nonlocal buf_paths, buf_texts, buf_len
        if buf_texts:
            path = _common_heading_path(buf_paths)
            chunks.append({"heading_path": path, "content": "\n\n".join(buf_texts)})
        buf_paths, buf_texts, buf_len = [], [], 0

    for heading_path, text in sections:
        if len(text) > max_chars:
            emit_buffer()  # bekleyen küçük bölümleri kapat
            for part in chunk_text(text, max_chars):  # dev bölümü alt-böl
                chunks.append({"heading_path": heading_path, "content": part})
            continue
        if buf_texts and buf_len + 2 + len(text) > max_chars:
            emit_buffer()  # max'ı aşacaksa önce mevcut chunk'ı kapat
        buf_paths.append(heading_path)
        buf_texts.append(text)
        buf_len += len(text) + 2
        if buf_len >= MIN_CHUNK_CHARS:
            emit_buffer()  # tabana ulaştı, gereksiz birleştirme yapma
    emit_buffer()
    return chunks


def _common_heading_path(paths: list[str]) -> str:
    """Birden fazla bölümün birleştiği chunk'ın başlık yolu = ortak en uzun ön ek.
    Ör. 'Doc > Install > Windows' + 'Doc > Install > Linux' → 'Doc > Install'."""
    if len(paths) == 1:
        return paths[0]
    seg_lists = [p.split(" > ") for p in paths]
    common: list[str] = []
    for segs in zip(*seg_lists):
        if all(s == segs[0] for s in segs):
            common.append(segs[0])
        else:
            break
    return " > ".join(common) if common else paths[0]


def main():
    print(f"Profil: {PROFILE}  ({DOCUMENTS_DIR}/ → {DB_PATH})")
    config = Configuration(app_name="LocalRagAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model bulunamadı: {EMBEDDING_MODEL_ALIAS}")
    if not model.is_cached:
        print(f"'{EMBEDDING_MODEL_ALIAS}' indiriliyor (ilk kullanım)...")
        model.download(progress_callback=lambda pct: print(f"  %{pct:.0f}", end="\r"))
        print()
    model.load()
    embed_client = model.get_embedding_client()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)  # DROP+CREATE → tablo boş başlar

    total_chunks = 0
    for path in sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt"))):
        filename = os.path.basename(path)
        source_url, body = read_document(path)
        chunks = chunk_markdown(body)
        print(f"{filename}: {len(chunks)} chunk")

        # Embedding için başlık yolu + içerik birlikte kullanılıyor, içerik ise ham saklanıyor
        # (LLM'e olduğu gibi verilecek). Doküman tarafını başlıkla zenginleştirip sorguyu ham
        # bırakmak, doğal soruyla teknik doküman arasındaki boşluğu arama sırasında kapatıyor.
        embed_texts = ["\n\n".join(p for p in (c["heading_path"], c["content"]) if p) for c in chunks]
        response = embed_client.generate_embeddings(embed_texts)

        for chunk, item in zip(chunks, response.data):
            cursor = conn.execute(
                "INSERT INTO documents (source, heading_path, content, embedding) VALUES (?, ?, ?, ?)",
                (filename, chunk["heading_path"], chunk["content"], json.dumps(item.embedding)),
            )
            conn.execute(
                "INSERT INTO documents_fts (rowid, content) VALUES (?, ?)",
                (cursor.lastrowid, chunk["content"]),
            )
        total_chunks += len(chunks)

    conn.commit()

    row_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"\nToplam {total_chunks} chunk üretildi, veritabanında {row_count} satır var.")
    assert total_chunks == row_count, "Chunk sayısı ile DB satır sayısı uyuşmuyor!"

    conn.close()


if __name__ == "__main__":
    main()
