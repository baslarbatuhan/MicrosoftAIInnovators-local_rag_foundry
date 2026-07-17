# Local RAG Q&A Assistant

Tamamen **çevrimdışı** çalışan, belge tabanlı soru-cevap asistanı. Microsoft **Foundry Local**
ile LLM'i cihaz üzerinde çalıştırır; **RAG** (Retrieval-Augmented Generation) deseniyle
cevaplarını yerel bilgi tabanındaki belgelere dayandırır — internet bağlantısı, bulut hesabı
veya GPU gerektirmez.

> Bu proje, Microsoft Tech Community'deki
> [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968)
> rehberinden ilham alan bir yaz okulu projesi olarak geliştirilmiştir.

## Özellikler

- 🔌 **%100 yerel:** Model indirme sonrası hiçbir ağ çağrısı yok
- 📚 **Kaynak gösterimi:** Her cevabın hangi belgeden geldiği deterministik olarak raporlanır
- 🚫 **Uydurma yok:** Cevap bilgi tabanında yoksa model tahmin etmez, "I don't have that information" der (değerlendirmede 10/10 doğru reddetme)
- 🔍 **Şeffaf retrieval:** `--verbose` bayrağıyla modelin gördüğü chunk'lar skorlarıyla izlenebilir

## Mimari

```
Kullanıcı Sorusu (CLI)
      │
      ▼
answer_query()  ──►  get_top_chunks()  ──►  SQLite (chunk + embedding)
      │                    │ cosine similarity, top-3
      │              qwen3-embedding-0.6b (soru embedding'i)
      ▼
phi-3.5-mini  ◄── system prompt + [Source] etiketli bağlam + soru
      │
      ▼
Cevap + Kaynak listesi
```

| Bileşen | Teknoloji |
|---|---|
| LLM runtime | Microsoft Foundry Local (in-process SDK, v1.2.3) |
| Chat modeli | `phi-3.5-mini` (3.8B, `temperature=0`, `max_tokens=256`) |
| Embedding modeli | `qwen3-embedding-0.6b` (1024 boyutlu vektörler) |
| Vektör deposu | SQLite (JSON-serileştirilmiş embedding'ler, brute-force cosine similarity) |
| Bilgi tabanı | 5 belge / 114 chunk (paragraf bazlı + overlap, max 800 karakter) |

## Kurulum

**Gereksinimler:** Windows, Python 3.10+ (3.14 ile test edildi)

```powershell
# 1. Foundry Local'ı kur
winget install Microsoft.FoundryLocal --accept-source-agreements
foundry --version   # doğrulama

# 2. Python bağımlılıklarını kur
#    (birden fazla Python varsa "pip install" yerine mutlaka "python -m pip" kullan)
python -m pip install -r requirements.txt

# 3. Bilgi tabanını hazırla (Kaggle'dan 5 belge + test soru setlerini indirir)
python prepare_dataset.py

# 4. Belgeleri chunk'la, embed et, veritabanına yaz (data/rag.db)
python ingest.py
```

> İlk çalıştırmada modeller otomatik indirilir (`qwen3-embedding-0.6b` ~600 MB,
> `phi-3.5-mini` ~2 GB) ve cache'lenir — sonraki çalıştırmalar çevrimdışıdır.

## Kullanım

```powershell
python main.py            # normal mod
python main.py --verbose  # retrieval'ın getirdiği chunk'ları skorlarıyla gösterir
```

Örnek oturum (bilgi tabanı İngilizce olduğu için sorular İngilizce sorulmalı):

```
Soru: What do keybullet kin drop?

Cevap: Keybullet Kin drop keys upon death.
Kaynaklar: bullet_kin.txt

Soru: How much health does the Mutant Bullet Kin have?

Cevap: I don't have that information.
```

Çıkmak için `exit`, `quit` veya `çık`.

**Kendi belgelerinizi kullanmak için:** `documents/` klasörüne `.txt` dosyalarınızı koyup
`python ingest.py` çalıştırın (ilk satır `Kaynak: <url>` formatındaysa kaynak olarak ayrıştırılır).

## Değerlendirme

`python evaluate.py` — `eval/` klasöründeki 20 soruyu (10 cevaplanabilir + 10 cevaplanamaz)
çalıştırır, metrikleri konsola ve `eval/eval_results.csv`'ye yazar.

Nihai sonuçlar (deterministik, `temperature=0`):

| Metrik | Sonuç |
|---|---|
| Retrieval isabeti | 9/10 (%90)* |
| Cevaplama oranı | 8/10 (%80) |
| Doğru reddetme (uydurma yok) | **10/10 (%100)** |
| Yanıt süresi | ort. 13.0 s (CPU) |

\* Tek kaçırılan sorunun cevap pasajı dataset'teki belge metninde hiç yer almıyor
(dataset kusuru) — sistemin bu soruyu reddetmesi doğru davranış.

## Proje Yapısı

```
main.py              → giriş noktası: answer_query() + CLI döngüsü
ingest.py            → belgeleri chunk'la + embed et + SQLite'a yaz
retrieval.py         → get_top_chunks(): cosine similarity ile top-k chunk
evaluate.py          → 20 soruluk otomatik değerlendirme + metrik raporu
prepare_dataset.py   → Kaggle dataset'inden bilgi tabanı ve test sorularını hazırlar
documents/           → bilgi tabanı (.txt belgeler)
eval/                → test soru setleri + değerlendirme sonuçları
exercises/           → geliştirme sürecindeki kavramsal demo scriptleri (pipeline'ın parçası değil)
data/                → üretilen veritabanları (ingest.py otomatik oluşturur)
```

## Bilinen Sınırlamalar

- **Gecikme:** CPU üzerinde soru başına ~13 sn (3.8B model gerçeği). Hedef donanımda GPU
  varyantı veya daha küçük model seçilerek düşürülebilir.
- **Küçük model okuma sınırı:** Model nadiren bağlamdaki *benzer* bir pasajı sorulanla
  karıştırabiliyor (ör. iki farklı changelog satırını karıştırdığı bir vaka gözlendi).
  Uydurma değildir — cevap her zaman bağlamdan gelir — ama doğruluğu garanti etmez.
- **Fazla temkinlilik:** Sıkı grounding, sınırdaki 1-2 cevaplanabilir soruyu da
  reddettirebiliyor (precision/recall takası — uydurmasızlık lehine bilinçli tercih).
- **Dil:** Bilgi tabanı İngilizce olduğu için prompt ve sorular İngilizce'dir. Türkçe belge
  koyarsanız system prompt'un da Türkçeleştirilmesi gerekir (`main.py` → `SYSTEM_PROMPT`).
- **Ölçek:** Brute-force cosine similarity küçük koleksiyonlar için yeterlidir; binlerce
  belgede özel vektör veritabanına (ChromaDB, Qdrant vb.) geçilmelidir.

## Öğrenilen Dersler

1. **Dil tutarlılığı kritik:** Türkçe system prompt + İngilizce içerik kombinasyonu modeli
   şaşırtıyordu; prompt'u İngilizceye çevirmek doğru reddetme oranını %30'dan %100'e çıkardı.
2. **Chunk sınırları retrieval kalitesini belirliyor:** Başlık/tarih satırlarının içerikten
   ayrı chunk'a düşmesi cevaplanabilir soruları kaybettiriyordu; chunk overlap bunu çözdü.
3. **Daha fazla bağlam ≠ daha iyi:** `top_k`'yı 3'ten 5'e çıkarmak cevaplamayı artırdı ama
   uydurmayı geri getirdi ve gecikmeyi 2.4× yaptı — geri alındı.
4. **Ölçmeden bilemezsin:** "Çalışıyor gibi görünen" ilk sistemde 4 gizli sorun vardı;
   hepsi ancak sistematik değerlendirmeyle (7 tur ölç→teşhis→düzelt) ortaya çıktı.

## Kaynaklar

- [Microsoft Foundry Local dokümantasyonu](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/)
- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) (Tech Community)
- Bilgi tabanı ve test soruları: [single-topic-rag-evaluation-dataset](https://www.kaggle.com/datasets/samuelmatsuoharris/single-topic-rag-evaluation-dataset) (Kaggle)
