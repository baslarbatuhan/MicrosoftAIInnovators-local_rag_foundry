"""
Multi-turn değerlendirme: multiturn_scenarios.json'daki senaryoları app.py/CLI ile aynı şekilde
(history tur tur birikerek) answer_query üzerinden çalıştırır ve tek-tur eval'in göremediği dört
davranışı ölçer:

  - coref retrieval isabeti: takip sorusu ('that', 'it') condense edildikten sonra doğru dokümana
    ulaştı mı? (on_retrieval callback'i modele giden gerçek sorgunun getirdiği kaynakları yakalar.)
  - topic-switch drift: yeni/bağımsız bir soruda condensation eski konuyu sürükleyip yanlış
    dokümana mı gitti?
  - grounding: cevabı korpusta olmayan takip sorusunda model reddetti mi, yoksa uydurdu mu?
    (expected_source boşsa reddetme bekleniyor.)
  - baseline: standalone turlar hâlâ doğru dokümanı buluyor mu?

Bu script pipeline'a dokunmuyor, sadece ölçüyor. `python -m rag.evaluate_multiturn`.
"""
import json
import os
import time

from rag.config import EVAL_DIR
from rag.core import answer_query, is_refusal, load_chat_client

SCENARIOS_PATH = os.path.join(EVAL_DIR, "multiturn_scenarios.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "multiturn_results.csv")


def load_scenarios() -> list[dict]:
    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        return json.load(f)["scenarios"]


def evaluate_turn(question: str, chat_client, history: list[dict], expected_source: str) -> dict:
    """Tek bir turu çalıştırıp teşhis alanlarını döner. on_retrieval ile condensation sonrası modele
    giden gerçek sorgu ve kaynaklar yakalanır; böylece model reddetse bile retrieval isabetini
    ayrıca ölçebiliyoruz."""
    captured: dict = {}

    def on_retrieval(search_query: str, chunks: list) -> None:
        captured["search_query"] = search_query
        captured["sources"] = [c["source"] for c in chunks]

    start = time.perf_counter()
    answer, _ = answer_query(question, chat_client, history=history, on_retrieval=on_retrieval)
    elapsed = time.perf_counter() - start

    refused = is_refusal(answer)
    retrieved = set(captured.get("sources", []))
    # expected_source '' → cevaplanamaz (reddetme beklenir); '|' ile alternatifler.
    if expected_source:
        hit = any(s.strip() in retrieved for s in expected_source.split("|"))
        correct = hit and not refused          # cevaplanabilir tur: doğru dokümanı bulup cevaplamalı
    else:
        hit = None
        correct = refused                       # cevaplanamaz tur: reddetmeli (grounding probu)

    return {
        "question": question, "search_query": captured.get("search_query", question),
        "condensed": captured.get("search_query", question) != question,
        "retrieved": "|".join(sorted(retrieved)), "expected_source": expected_source,
        "retrieval_hit": hit, "refused": refused, "correct": correct,
        "latency_s": round(elapsed, 2), "answer": answer,
    }


def main():
    print("Model yükleniyor...")
    chat_client = load_chat_client()

    scenarios = load_scenarios()
    rows: list[dict] = []
    by_kind: dict[str, list[bool]] = {}

    for scen in scenarios:
        print(f"\n=== {scen['name']} ===")
        history: list[dict] = []
        for turn in scen["turns"]:
            kind = turn.get("kind", "turn")
            res = evaluate_turn(turn["question"], chat_client, history, turn.get("expected_source", ""))
            res["scenario"] = scen["name"]
            res["kind"] = kind
            rows.append(res)
            by_kind.setdefault(kind, []).append(res["correct"])

            flag = "OK  " if res["correct"] else "FAIL"
            hit_text = "hit" if res["retrieval_hit"] else ("miss" if res["retrieval_hit"] is False else "----")
            cond = f"  ~> '{res['search_query']}'" if res["condensed"] else ""
            print(f"  [{flag}] {res['latency_s']:4.1f}s  {hit_text}  refused={int(res['refused'])}  "
                  f"{turn['question'][:48]}{cond}")

            # Sonraki tura geçmiş olarak gerçek cevabı ekliyoruz (app.py/CLI ile aynı akış).
            history.append({"question": turn["question"], "answer": res["answer"]})

    print("\n" + "=" * 56)
    print("ÖZET (kind bazında doğru / toplam)")
    print("=" * 56)
    for kind, flags in by_kind.items():
        print(f"  {kind:20s}: {sum(flags)}/{len(flags)}")
    total = [f for flags in by_kind.values() for f in flags]
    print(f"  {'TOPLAM':20s}: {sum(total)}/{len(total)}")

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDetaylı sonuçlar: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
