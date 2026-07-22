"""
Hafta 5 değerlendirme scripti: eval/ klasöründeki hazır soru setlerini
answer_query() üzerinden çalıştırır ve şu metrikleri ölçer:

  Cevaplanabilir sorular (single_passage_answer_questions.csv):
    - retrieval isabeti: beklenen kaynak dosya top_k chunk'lara geldi mi?
    - cevaplama: model reddetmeyip cevap verdi mi?
    - cevap doğruluğu: model cevabı, dataset'teki referans cevaba semantik
      olarak benziyor mu? (embedding cosine similarity >= SIM_THRESHOLD)
  Cevaplanamaz sorular (no_answer_questions.csv):
    - doğru reddetme: model "Bu bilgiye sahip değilim." dedi mi?
  Performans: her soru için yanıt süresi (embedding + retrieval + generation).

Sonuçlar konsola özet olarak yazılır ve eval/eval_results.csv'ye kaydedilir.
"""
import csv
import os
import re
import time

from rag.config import EVAL_DIR, TOP_K
from rag.core import answer_query, load_chat_client, is_refusal
from rag.retrieval import EMBEDDING_MODEL_ALIAS, cosine_similarity, get_manager, get_top_chunks

RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results.csv")

# Cevap doğruluğu eşiği: recall-odaklı max-sim (answer_similarity) bu değerin üstündeyse "doğru".
# YENİDEN KALİBRE (2026-07-22, whole-text cosine → max-sim recall geçişiyle): max-sim skorları
# cosine'den yüksek çıkar (en iyi cümle eşleşmesi), eski 0.65 transfer olmaz. Config-5 cevaplarında
# gözlenen dağılım: bilinen-NEGATİF (inference-engine off-target) = 0.607 | en-düşük-doğru (SDK 0.8.0
# zengin/kod-bloklu, eski cosine'de 0.516 alıp YANLIŞ sayılıyordu) = 0.656 | net-doğrular 0.79-0.97.
# 0.63 = negatifi (0.607) eleyip tüm doğruları (≥0.656) geçiren nokta. Not: tek negatifle bant dar
# (~0.05); soru/negatif seti büyürse yeniden gözden geçirilmeli. Bu geçiş SDK 0.8.0 bias'ını çözdü.
SIM_THRESHOLD = 0.63


def load_questions(filename: str) -> list[dict]:
    with open(os.path.join(EVAL_DIR, filename), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_embed_client():
    """Cevap doğruluğu ölçümü için embedding client (retrieval ile aynı model)."""
    manager = get_manager()
    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)
    model.load()
    return model.get_embedding_client()


def split_sentences(text: str) -> list[str]:
    """Metni cümlelere böler. Fenced kod blokları (```...```) TEK atomik birim olarak
    korunur (içindeki noktalardan bölünmez — kod bir 'iddia'dır). Kalan düzyazı cümle
    sonlarından (. ! ?) ayrılır. Çok kısa parçalar (< 4 char) elenir."""
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("```"):
            sentences.append(part)  # kod bloğu: atomik
        else:
            for s in re.split(r"(?<=[.!?])\s+", part):
                s = s.strip()
                if len(s) > 3:
                    sentences.append(s)
    return sentences


def answer_similarity(embed_client, model_answer: str, reference_answer: str) -> float:
    """Model cevabının referansı ne kadar KAPSADIĞI — recall-odaklı max-similarity (BERTScore mantığı).

    ❌ ESKİ: `cosine(embed(tüm_cevap), embed(referans))` — bütün-metin cosine, ZENGİN cevabı
    cezalandırıyordu: kod-bloklu/detaylı bir cevabın embedding'i tek vektöre sıkışıp (dilution) terse
    referanstan uzaklaşıyordu (ör. SDK 0.8.0 kod-bloklu mükemmel cevap 0.516 aldı — yanlış sayıldı).

    ✅ YENİ: referansın her cümlesi için, cevabın cümleleri üzerinde MAX cosine al, ortalamasını dön.
    "Referansın içeriği cevabın BİR yerinde geçiyor mu?" → zengin cevap en iyi cümlesiyle eşleşir,
    ekstra cümleler skoru DÜŞÜRMEZ (dilution biter). Endüstri karşılığı: BERTScore-recall / ColBERT MaxSim.
    Aynı embedding modelini kullanır (yeni bağımlılık yok, deterministik). Precision/uydurma zaten
    reddetme 8/8 + grounded üretimle güvence altında; bu metrik eksik parçayı (recall) kapatır.
    """
    ans_sentences = split_sentences(model_answer)
    ref_sentences = split_sentences(reference_answer)
    if not ans_sentences or not ref_sentences:
        return 0.0
    response = embed_client.generate_embeddings(ans_sentences + ref_sentences)
    embeddings = [d.embedding for d in response.data]
    ans_embs = embeddings[:len(ans_sentences)]
    ref_embs = embeddings[len(ans_sentences):]
    per_ref = [max(cosine_similarity(r, a) for a in ans_embs) for r in ref_embs]
    return sum(per_ref) / len(per_ref)


def main():
    print("Model yükleniyor...")
    chat_client = load_chat_client()
    embed_client = get_embed_client()

    results = []
    latencies = []

    # --- Cevaplanabilir sorular ---
    answerable = load_questions("single_passage_answer_questions.csv")
    retrieval_hits = 0
    answered = 0
    correct_answers = 0
    print(f"\n=== Cevaplanabilir sorular ({len(answerable)}) ===")
    for row in answerable:
        question = row["question"]
        expected_source = row["source_file"]

        start = time.perf_counter()
        answer, sources = answer_query(question, chat_client)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        # Retrieval isabetini generation'dan bağımsız ölç: model reddetse bile
        # (sources=[] dönse bile) retrieval doğru dokümanı bulmuş olabilir.
        # source_file '|' ile alternatifler içerebilir (aynı bilgi birden çok dosyada).
        retrieved = get_top_chunks(question, top_k=TOP_K)
        retrieved_set = {c["source"] for c in retrieved}
        hit = any(s.strip() in retrieved_set for s in expected_source.split("|"))
        refused = is_refusal(answer)
        if hit:
            retrieval_hits += 1
        if not refused:
            answered += 1

        # Cevap doğruluğu: sadece cevaplanmış sorularda ölçülür (reddetmede anlamsız)
        similarity = None
        if not refused:
            similarity = answer_similarity(embed_client, answer, row["answer"])
            if similarity >= SIM_THRESHOLD:
                correct_answers += 1

        sim_text = f"sim={similarity:.2f}" if similarity is not None else "sim=----"
        print(f"  [{'OK ' if hit else 'MISS'}] {elapsed:4.1f}s  {sim_text}  {question[:50]}")
        results.append({
            "type": "answerable", "question": question, "expected_source": expected_source,
            "retrieved_sources": "|".join(sources), "retrieval_hit": hit,
            "refused": refused,
            "answer_similarity": round(similarity, 3) if similarity is not None else "",
            "latency_s": round(elapsed, 2), "answer": answer,
        })

    # --- Cevaplanamaz sorular ---
    unanswerable = load_questions("no_answer_questions.csv")
    correct_refusals = 0
    print(f"\n=== Cevaplanamaz sorular ({len(unanswerable)}) ===")
    for row in unanswerable:
        question = row["question"]

        start = time.perf_counter()
        answer, sources = answer_query(question, chat_client)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        refused = is_refusal(answer)
        if refused:
            correct_refusals += 1

        print(f"  [{'OK  ' if refused else 'FAIL'}] {elapsed:4.1f}s  {question[:55]}")
        results.append({
            "type": "unanswerable", "question": question, "expected_source": "",
            "retrieved_sources": "|".join(sources), "retrieval_hit": "",
            "refused": refused, "answer_similarity": "",
            "latency_s": round(elapsed, 2), "answer": answer,
        })

    # --- Özet ---
    n_ans = len(answerable)
    n_unans = len(unanswerable)
    print("\n" + "=" * 50)
    print("ÖZET")
    print("=" * 50)
    print(f"Retrieval isabeti (cevaplanabilir): {retrieval_hits}/{n_ans} = %{100*retrieval_hits/n_ans:.0f}")
    print(f"Cevaplama oranı  (cevaplanabilir): {answered}/{n_ans} = %{100*answered/n_ans:.0f}")
    print(f"Cevap doğruluğu  (sim>={SIM_THRESHOLD}, cevaplananlar içinde): {correct_answers}/{answered}")
    print(f"Doğru reddetme   (cevaplanamaz):   {correct_refusals}/{n_unans} = %{100*correct_refusals/n_unans:.0f}")
    print(f"Yanıt süresi: ort {sum(latencies)/len(latencies):.1f}s  |  min {min(latencies):.1f}s  |  max {max(latencies):.1f}s")

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetaylı sonuçlar: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
