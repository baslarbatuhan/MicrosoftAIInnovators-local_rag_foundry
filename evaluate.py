"""
Hafta 5 değerlendirme scripti: eval/ klasöründeki hazır soru setlerini
answer_query() üzerinden çalıştırır ve şu metrikleri ölçer:

  Cevaplanabilir sorular (single_passage_answer_questions.csv):
    - retrieval isabeti: beklenen kaynak dosya top_k chunk'lara geldi mi?
    - cevaplama: model reddetmeyip cevap verdi mi?
  Cevaplanamaz sorular (no_answer_questions.csv):
    - doğru reddetme: model "Bu bilgiye sahip değilim." dedi mi?
  Performans: her soru için yanıt süresi (embedding + retrieval + generation).

Sonuçlar konsola özet olarak yazılır ve eval/eval_results.csv'ye kaydedilir.
"""
import csv
import os
import time

from main import answer_query, load_chat_client, is_refusal
from retrieval import get_top_chunks

EVAL_DIR = "eval"
RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results.csv")


def load_questions(filename: str) -> list[dict]:
    with open(os.path.join(EVAL_DIR, filename), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("Model yükleniyor...")
    chat_client = load_chat_client()

    results = []
    latencies = []

    # --- Cevaplanabilir sorular ---
    answerable = load_questions("single_passage_answer_questions.csv")
    retrieval_hits = 0
    answered = 0
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
        # top_k, answer_query'nin varsayılanıyla aynı tutulmalı (LLM'in gördüğü bağlam).
        retrieved = get_top_chunks(question, top_k=3)
        hit = expected_source in {c["source"] for c in retrieved}
        refused = is_refusal(answer)
        if hit:
            retrieval_hits += 1
        if not refused:
            answered += 1

        print(f"  [{'OK ' if hit else 'MISS'}] {elapsed:4.1f}s  {question[:55]}")
        results.append({
            "type": "answerable", "question": question, "expected_source": expected_source,
            "retrieved_sources": "|".join(sources), "retrieval_hit": hit,
            "refused": refused, "latency_s": round(elapsed, 2), "answer": answer,
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
            "refused": refused, "latency_s": round(elapsed, 2), "answer": answer,
        })

    # --- Özet ---
    n_ans = len(answerable)
    n_unans = len(unanswerable)
    print("\n" + "=" * 50)
    print("ÖZET")
    print("=" * 50)
    print(f"Retrieval isabeti (cevaplanabilir): {retrieval_hits}/{n_ans} = %{100*retrieval_hits/n_ans:.0f}")
    print(f"Cevaplama oranı  (cevaplanabilir): {answered}/{n_ans} = %{100*answered/n_ans:.0f}")
    print(f"Doğru reddetme   (cevaplanamaz):   {correct_refusals}/{n_unans} = %{100*correct_refusals/n_unans:.0f}")
    print(f"Yanıt süresi: ort {sum(latencies)/len(latencies):.1f}s  |  min {min(latencies):.1f}s  |  max {max(latencies):.1f}s")

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetaylı sonuçlar: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
