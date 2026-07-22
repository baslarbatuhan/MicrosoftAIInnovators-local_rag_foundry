"""
Retrieval pipeline (get_top_chunks):
  1. Sorguyu embed et, tüm chunk'lara karşı cosine similarity → vektör top-N aday.
  2. HYBRID_RETRIEVAL açıksa FTS5/BM25 keyword adaylarını da havuza ekle (kesin terimleri
     — ONNX, komut adları — yakalar; vektörün kaçırdığını sokar).
  3. RERANK açıksa cross-encoder birleşik havuzu (soru, chunk) çifti olarak yeniden puanlar → top_k.

Kritik tasarım: ham BM25 modele DEĞİL reranker'a beslenir. Ham BM25→model füzyonu (RRF) 2× ölçülüp
reddedilmişti (keyword-eşleşen alakasız bağlam uydurma tetikliyordu); reranker o güvenlik filtresini
sağlıyor. expand_parents(): retrieved chunk'ı parent-document bağlamına genişletir (bkz. main.py).
"""
import json
import re
import sqlite3

from foundry_local_sdk import Configuration, FoundryLocalManager

from rag import reranker
from rag import telemetry
from rag.config import DB_PATH, EMBEDDING_MODEL_ALIAS, HYBRID_RETRIEVAL, RERANK, RERANK_CANDIDATES

FUSION_CANDIDATES = 20  # HYBRID açıkken BM25'ten çekilip reranker havuzuna eklenen aday sayısı

# BM25 sorgusundan elenen İngilizce stopword'ler — OR sorgusunda kalırlarsa
# neredeyse her chunk eşleşir ve sıralama kirlenir (ölçümle görüldü).
STOPWORDS = frozenset(
    "a an and are as at be by can do does did for from had has have how i in is it its "
    "of on or that the this to was were what when where which who why will with you your".split()
)


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


def _bm25_ranking(conn: sqlite3.Connection, query: str, limit: int) -> list[int]:
    """FTS5/BM25 keyword sıralaması — en alakalıdan aza doğru chunk id listesi.

    Soru, FTS5 sorgu sözdizimine takılmasın diye sade alfanümerik token'lara
    indirgenir; stopword'ler elenir; kalanlar OR ile birleştirilir.
    """
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    tokens = [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]
    if not tokens:
        return []

    rows = conn.execute(
        "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? "
        "ORDER BY bm25(documents_fts) LIMIT ?",  # bm25(): düşük değer = daha alakalı
        (" OR ".join(tokens), limit),
    ).fetchall()
    return [row_id for (row_id,) in rows]


def get_top_chunks(query: str, top_k: int = 3) -> list[dict]:
    """Verilen sorgu için en alakalı top_k chunk'ı döner.
    Her sonuç {'id', 'source', 'heading_path', 'content', 'score'} (+ RERANK açıksa 'rerank_score').
    'score' = cosine similarity; nihai sıralama reranker skorundadır."""
    manager = get_manager()
    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    model.load()
    embed_client = model.get_embedding_client()

    with telemetry.stage("embed"):
        query_embedding = embed_client.generate_embedding(query).data[0].embedding

    with telemetry.stage("search"):
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT id, source, heading_path, content, embedding FROM documents").fetchall()
        bm25_ids = _bm25_ranking(conn, query, FUSION_CANDIDATES) if HYBRID_RETRIEVAL else []
        conn.close()

        scored = {
            row_id: {"id": row_id, "source": source, "heading_path": heading_path, "content": content,
                     "score": cosine_similarity(query_embedding, json.loads(embedding_json))}
            for row_id, source, heading_path, content, embedding_json in rows
        }
        vector_ids = sorted(scored, key=lambda i: scored[i]["score"], reverse=True)

    if RERANK:
        # Aday havuzu: vektör yüksek-recall top-N. HYBRID açıksa BM25 adayları da eklenir
        # (kesin terimleri — ONNX, komut adları — yakalar; vektörün kaçırdığını havuza sokar).
        # Reranker bu BİRLEŞİK havuzu (soru, chunk) çifti olarak yeniden puanlar → en iyi top_k.
        # Kritik: ham BM25'i modele DEĞİL reranker'a besliyoruz — eski hybrid redlerinde
        # eksik olan güvenlik filtresi bu (keyword-eşleşen alakasız bağlam reranker'da elenir).
        cand_ids = list(vector_ids[:RERANK_CANDIDATES])
        for row_id in bm25_ids:  # HYBRID kapalıysa bm25_ids == [] → salt-vektör havuzu
            if row_id not in cand_ids:
                cand_ids.append(row_id)
        candidates = [scored[i] for i in cand_ids]
        with telemetry.stage("rerank"):
            return reranker.rerank(query, candidates, top_k)

    # RERANK kapalı (ablation/debug) → salt-vektör top_k. Hibrit BM25 YALNIZCA reranker havuzunu
    # besler; ham BM25→model füzyonu (RRF) 2× ölçülüp reddedildi ve kaldırıldı (bkz. PLAN.md).
    return [scored[i] for i in vector_ids[:top_k]]


def _window(chunks_in_doc: list[tuple[int, str]], center_id, cap: int) -> str:
    """Bir dokümanın chunk içeriklerini id sırasında birleştirir. Toplam cap'i aşarsa,
    retrieved chunk'a (center_id) ortalanmış bir pencere alır (iki yöne dengeli büyüyerek) —
    böylece büyük dokümanlarda bile en alakalı bölge + komşuları modele gider."""
    full = "\n\n".join(content for _, content in chunks_in_doc)
    if len(full) <= cap:
        return full
    idx = next((k for k, (i, _) in enumerate(chunks_in_doc) if i == center_id), 0)
    picked = [chunks_in_doc[idx][1]]
    size, lo, hi = len(picked[0]), idx, idx
    while True:
        grew = False
        if lo - 1 >= 0 and size + len(chunks_in_doc[lo - 1][1]) + 2 <= cap:
            lo -= 1; picked.insert(0, chunks_in_doc[lo][1]); size += len(chunks_in_doc[lo][1]) + 2; grew = True
        if hi + 1 < len(chunks_in_doc) and size + len(chunks_in_doc[hi + 1][1]) + 2 <= cap:
            hi += 1; picked.append(chunks_in_doc[hi][1]); size += len(chunks_in_doc[hi][1]) + 2; grew = True
        if not grew:
            break
    return "\n\n".join(picked)


def expand_parents(chunks: list[dict], max_chars: int = None) -> list[dict]:
    """Parent-document retrieval: her retrieved chunk'ı ait olduğu DOKÜMANA genişletir
    (aynı source'un tüm chunk'ları, id sırasında; cap'i aşarsa retrieved chunk'a ortalı pencere).
    source bazında dedup (aynı dokümandan birden çok chunk gelirse tek parent). Sıralama korunur.

    Küçük dokümanlar (çoğu) TAM verilir → model "vizyonunu" kaybetmez, cevap komşu bölümdeyse de
    yakalar (section-level bunu ıskalıyordu). Büyük dokümanlar cap ile pencerelenir (gürültü/gecikme
    koruması). content genişler; source/heading_path korunur. Retrieval sırası/metrikleri etkilenmez."""
    from rag.config import PARENT_MAX_CHARS
    cap = PARENT_MAX_CHARS if max_chars is None else max_chars

    conn = sqlite3.connect(DB_PATH)
    by_source: dict[str, list[tuple[int, str]]] = {}
    result: list[dict] = []
    seen: set[str] = set()
    for ch in chunks:
        src = ch["source"]
        if src in seen:
            continue
        seen.add(src)
        if src not in by_source:
            by_source[src] = conn.execute(
                "SELECT id, content FROM documents WHERE source=? ORDER BY id", (src,)
            ).fetchall()
        content = _window(by_source[src], ch.get("id"), cap)
        result.append({"id": ch.get("id"), "source": src, "heading_path": ch.get("heading_path", ""),
                       "content": content, "score": ch.get("score"), "rerank_score": ch.get("rerank_score")})
    conn.close()
    return result


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
