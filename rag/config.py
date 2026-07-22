"""
Yapılandırma: pipeline hangi doküman klasörü + veritabanı + eval seti üçlüsüyle çalışacak,
hangi modeller ve retrieval bileşenleri aktif. Yollar repo köküne göre MUTLAK üretilir
(CWD'den bağımsız — paket nereden çalıştırılırsa çalıştırılsın doğru yolları bulur).
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # rag/ bir seviye altta → repo kökü

PROFILE = "foundry"
DOCS_DIR = str(_ROOT / "knowledge_bases" / "foundry" / "documents")
DB_PATH = str(_ROOT / "data" / "foundry.db")
EVAL_DIR = str(_ROOT / "knowledge_bases" / "foundry" / "eval")

# Embedding modeli — retrieval'ın kalbi. Foundry Local kataloğunda iki seçenek var:
#   qwen3-embedding-0.6b (1024 boyut, hızlı) — taban çizgisi, KULLANILAN
#   qwen3-embedding-8b   (5.6 GB, sadece CPU) — Deney A denendi ve BIRAKILDI: her sorguya
#                        kalıcı gecikme ekliyordu (GPU varyantı yok), indirme de takıldı.
# Değiştirirsen ingest.py YENİDEN çalıştırılmalı (sorgu+doküman vektörleri aynı modelden).
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

# Chat modeli — okuma + akıl yürütme + üretim. GPU varyantı olan modeller tercih edilir.
#   phi-3.5-mini (3.8B) — VARSAYILAN: hızlı (~1.5s), mevcut eval'de tavan kalite
#   qwen3-8b     (~5.5GB GPU) — Deney B denendi: temiz çalışıyor (/no_think ile) ama
#                mevcut "kolay" eval'de phi ile AYNI metrik (9/9/8), sadece 1.8× yavaş.
#                Farkını görmek için zorlu (doğal/sentez) sorular gerekiyor → eval genişletiliyor.
#                Altyapı hazır: bu satırı "qwen3-8b" yap + reasoning temizleme main.py'de otomatik.
CHAT_MODEL_ALIAS = "phi-3.5-mini"

# Reranking (cross-encoder): vektör retrieval yüksek-recall aday havuzu getirir
# (RERANK_CANDIDATES), bir cross-encoder bunları (soru, chunk) çifti olarak yeniden
# puanlayıp en iyi top_k'yı LLM'e verir. "Doğru chunk top-3'e giremiyor" sorununu çözer +
# hibrit BM25'i güvenli hale getiren filtre budur (ham füzyon uydurma tetikliyordu).
# Model: Xenova/bge-reranker-base ONNX (MIT), onnxruntime-CPU, PyTorch gerekmez. Bkz. reranker.py.
RERANK = True
RERANK_MODEL = "Xenova/bge-reranker-base"
RERANK_ONNX_FILE = "onnx/model_quantized.onnx"  # int8, ~280MB, CPU'da ~12ms/aday
RERANK_CANDIDATES = 15  # vektörden çekilen aday sayısı (recall); reranker bunları eler

# LLM'e verilen nihai chunk sayısı (reranker'ın döndürdüğü top_k).
# ❌ top_k=5 DENENDİ ve REDDEDİLDİ (2026-07-22, 37-soru eval): reddetme 11/11→10/11 (KAPI KIRILDI —
# "acquire company" trap'ine "Microsoft acquired Hugging Face" UYDURDU; 5 parent-doküman = fazla context
# → küçük model keyword-yakın içerikten fabrike etti) + gecikme 5.6→6.3s. Gerçek kalite kazancı YOK
# (correctness zaten 26/26; retrieval-hit +1 kozmetik — o soru top_k=3'te de doğru cevaplanıyordu).
# Ders: parent-doc zaten context'i genişletiyor; top_k'yi de artırmak flooding'i katlayıp uydurma tetikliyor.
TOP_K = 3

# Parent-document retrieval: retrieval küçük chunk üzerinden İSABETLİ arar, ama modele cevap
# üretirken o chunk'ın ait olduğu DOKÜMANI (aynı source'un chunk'ları, id sırasında; cap'i
# aşarsa retrieved chunk'a ortalı pencere) verir → model "vizyonunu" kaybetmez, terse/fragment
# chunk'lar tam bağlamla anlam kazanır. Sıralama/metrik değişmez; sadece LLM'in gördüğü CONTENT genişler.
# NOT (2026-07-22): cap=3000 doğruluğu düzeltti (format 0.505→0.73, doğruluk 16/16) AMA fazla context
# küçük modeli reddetme disiplininden çıkardı → codename sorusuna uydurma (reddetme 7/8, KAPI KIRILDI).
# Çözüm: cap'i DÜŞÜRMEDİK (kapsamı feda etmemek için); onun yerine SYSTEM_PROMPT sertleştirildi
# (bkz. main.py — anti-deflection kuralı). Cap yüksek kalıp reddetme prompt'la geri kazanılıyor.
PARENT_DOC = True
PARENT_MAX_CHARS = 3000  # doküman başına üst sınır; aşarsa retrieved chunk'a ortalı pencere alınır

# Hibrit retrieval (FTS5/BM25): kesin terimleri (ONNX, komut adları) yakalayıp vektörün
# kaçırdığı chunk'ları reranker ADAY HAVUZUNA sokar. Geçmişte 2× reddedilmişti — AMA o zaman
# BM25 ham füzyonla (RRF) doğrudan MODELE gidiyordu ve keyword-eşleşen alakasız bağlam uydurma
# tetikliyordu. Artık reranker (RERANK) güvenlik filtresi olarak arada: BM25 sadece aday üretir,
# reranker eler. Bu, redderken işaretlediğimiz "sadece reranker'la birlikte dene" koşulu.
# NOT: RERANK=True iken etkilidir (havuza BM25 adayı ekler). RERANK=False'a düşürülmemeli
# (ham BM25 füzyonu 2× ölçülüp reddedildi — bkz. PLAN.md). Değişiklikler eval'de ölçülür.
HYBRID_RETRIEVAL = True

# Not: Query-rewrite ve Possible-Questions denendi, ölçüldü ve REDDEDİLDİ (kod kaldırıldı;
# gerekçe + sayılar PLAN.md "Ölçülüp REDDEDİLEN denemeler" ve memory'de belgeli).
