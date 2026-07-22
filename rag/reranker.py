"""
Cross-encoder reranker (yerel, ücretsiz, offline).

Vektör retrieval yüksek-recall aday havuzu getirir (ör. top-15); bu modül o adayları
bir cross-encoder ile (soru, chunk) çifti olarak GERÇEK alaka düzeyine göre yeniden
puanlayıp en iyi top_k'yı seçer. Bi-encoder (embedding) sorgu ile chunk'ı ayrı ayrı
kodlar; cross-encoder ikisini BİRLİKTE işler → çok daha isabetli alaka skoru.

Model: BAAI bge-reranker-base'in ONNX'e çevrilmiş hali (Xenova/bge-reranker-base, MIT).
Runtime: onnxruntime (CPU) — PyTorch GEREKMEZ (Python 3.14 uyumu bu yüzden sorunsuz).
Model bir kez HuggingFace'ten indirilir (~280MB int8) ve cache'lenir; sonrası offline.

Chunk'lar `heading_path + content` ile puanlanır (embedding ile aynı asimetri) — terse
kod/komut chunk'ları bölüm bağlamıyla (ör. "Install packages") daha doğru skor alsın diye.
"""
from rag import config

_session = None
_tokenizer = None


def _load():
    """ONNX oturumunu ve tokenizer'ı bir kez yükler (lazy singleton). Ağır import'lar
    (onnxruntime, tokenizers) sadece reranker gerçekten kullanılınca yapılır — böylece
    RERANK kapalıyken ve testlerde bu bağımlılıklar hiç gerekmez."""
    global _session, _tokenizer
    if _session is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort

        onnx_path = hf_hub_download(config.RERANK_MODEL, config.RERANK_ONNX_FILE)
        tok_path = hf_hub_download(config.RERANK_MODEL, "tokenizer.json")
        _session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _tokenizer = Tokenizer.from_file(tok_path)
        _tokenizer.enable_truncation(max_length=512)
        _tokenizer.enable_padding()
    return _session, _tokenizer


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Adayları (soru, heading_path+content) alaka skoruna göre yeniden sıralar, top_k döner.
    Her chunk'a 'rerank_score' ekler (yüksek = daha alakalı). chunks boşsa aynen döner."""
    if not chunks:
        return chunks
    import numpy as np

    session, tokenizer = _load()
    pairs = [(query, f"{c['heading_path']}\n{c['content']}" if c.get("heading_path") else c["content"])
             for c in chunks]
    encs = tokenizer.encode_batch(pairs)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    if any(i.name == "token_type_ids" for i in session.get_inputs()):
        feed["token_type_ids"] = np.zeros_like(ids)

    logits = session.run(None, feed)[0].reshape(-1)
    for chunk, score in zip(chunks, logits):
        chunk["rerank_score"] = float(score)
    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    return chunks[:top_k]
