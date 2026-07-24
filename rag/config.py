"""
Pipeline ayarları: hangi doküman klasörü, veritabanı ve eval seti kullanılacak,
hangi modeller ve retrieval bileşenleri açık. Yollar repo köküne göre mutlak üretilir,
böylece paket nereden çalıştırılırsa çalıştırılsın doğru yerleri bulur.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # rag/ bir seviye altta, yani burası repo kökü

PROFILE = "foundry"
DOCS_DIR = str(_ROOT / "knowledge_bases" / "foundry" / "documents")
DB_PATH = str(_ROOT / "data" / "foundry.db")
EVAL_DIR = str(_ROOT / "knowledge_bases" / "foundry" / "eval")

# Embedding modeli. 0.6b sürümünü kullanıyoruz; 8b sürümünün GPU varyantı yok, her sorguya
# gecikme ekliyordu. Bunu değiştirirsen ingest.py'yi yeniden çalıştır (sorgu ve doküman
# vektörleri aynı modelden gelmeli).
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

# Chat modeli. phi-3.5-mini hızlı ve mevcut eval'de yeterli. qwen3-8b da çalışıyor ama daha
# yavaş ve kolay sorularda fark etmiyor; geçmek istersen /no_think temizliği core.py'de hazır.
CHAT_MODEL_ALIAS = "phi-3.5-mini"

# Reranker (cross-encoder). Vektör araması geniş bir aday havuzu getiriyor, reranker bunları
# (soru, chunk) çifti olarak yeniden puanlayıp en iyi top_k'yı seçiyor. Doğru chunk vektör
# top-3'e giremediğinde onu yukarı çekiyor; ayrıca BM25 gürültüsünü eleyen güvenlik filtresi bu.
RERANK = True
RERANK_MODEL = "Xenova/bge-reranker-base"
RERANK_ONNX_FILE = "onnx/model_quantized.onnx"  # int8, ~280MB, CPU'da aday başına ~12ms
RERANK_CANDIDATES = 15  # vektörden çekilen aday sayısı; reranker bunları eleyip top_k'ya indirir

# Modele kaç chunk gidecek. 3'te bırakıyoruz; 5'e çıkarınca fazla bağlam küçük modeli tuzak
# sorularda uydurmaya itti. Parent-doc zaten bağlamı genişletiyor, üstüne çıkmak gereksiz.
TOP_K = 3

# Parent-document retrieval: arama küçük chunk üzerinden isabetli yapılır, ama modele o chunk'ın
# ait olduğu tüm doküman verilir (cap'i aşarsa chunk'a ortalanmış bir pencere). Böylece kısa/kopuk
# chunk'lar çevresindeki bağlamla anlam kazanıyor. Bağlam büyüdükçe model reddetmeyi bırakabiliyor,
# onu system prompt'taki grounding kurallarıyla dengeliyoruz (cap'i düşürmek yerine).
PARENT_DOC = True
PARENT_MAX_CHARS = 3000  # doküman başına üst sınır; aşarsa chunk'a ortalanmış pencere alınır

# condense_query, takip sorusunu bağımsızlaştırırken son N turu bağlam olarak okur (coreference için).
# Pencereyi küçük tutuyoruz. Not: geçmiş yalnızca retrieval'a (condensation) gidiyor, generation'a
# değil — önceki cevapları prompt'a koymak yanlış atıflara yol açıyordu, o yüzden kaldırdık.
HISTORY_TURNS = 2

# Takip sorusunu retrieval'dan önce bağımsız sorguya çevir ("What format does it need?" + geçmiş
# → "What format does a Hugging Face model need?"). Ham takip sorusu tek başına yanlış chunk
# getiriyordu. Reddettiğimiz query-expansion'dan farklı: korpus terimi değil, konuşmadaki bilgiyi
# kullanıyor. Sadece history varken çalışır, o yüzden tek-turlu eval'i etkilemez.
CONDENSE_HISTORY = True

# Hibrit retrieval: BM25 ile kesin terimleri (komut adları, ONNX) yakalayıp reranker'ın aday
# havuzuna ekler. Ham BM25'i doğrudan modele vermek eskiden uydurma tetikliyordu; reranker
# gürültüyü elediği için RERANK açıkken güvenli, o yüzden RERANK=False ile birlikte kullanma.
HYBRID_RETRIEVAL = True

# Query-rewrite ve possible-questions denendi ama işe yaramadı, kaldırıldı. Gerekçeleri PLAN.md'de.
