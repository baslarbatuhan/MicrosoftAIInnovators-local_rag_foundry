"""
Hafif telemetri: her sorgunun aşama sürelerini ve meta verisini data/telemetry.jsonl'a
bir JSON satırı olarak ekler.

Amaç: gecikme bozulunca "nerede yavaşladı?" sorusunu her zaman kayıttan cevaplayabilmek.
Aşamaları bu şekilde profilleyip GPU'ya geçmeye de karar vermiştik.

Kullanım (pipeline içinden):
    telemetry.start_query(question)          # answer_query başında
    with telemetry.stage("embed"): ...       # aşamalar kendi süresini yazar
    telemetry.finish_query(refused=..., ...) # kaydı diske yazar

Aktif kayıt yokken stage()/add() sessizce hiçbir şey yapmaz, böylece get_top_chunks() tek
başına çağrıldığında çift kayıt oluşmuyor.

Kapatmak için: RAG_TELEMETRY=0
Özet rapor:   python telemetry.py
"""
import json
import os
import threading
import time

from rag.config import PROFILE

TELEMETRY_PATH = os.path.join("data", "telemetry.jsonl")
ENABLED = os.environ.get("RAG_TELEMETRY", "1") != "0"

_local = threading.local()

# Son tamamlanan sorgunun kaydı — CLI (--verbose) ve Streamlit, answer_query'nin
# dönüş imzasını değiştirmeden süreleri gösterebilsin diye.
last_record: dict | None = None


def start_query(question: str) -> None:
    if not ENABLED:
        return
    _local.record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "profile": PROFILE,
        "question_chars": len(question),
    }
    _local.t0 = time.perf_counter()


def add(key: str, value) -> None:
    """Aktif kayda alan ekler; aktif kayıt yoksa sessizce geçer."""
    record = getattr(_local, "record", None)
    if record is not None:
        record[key] = value


class stage:
    """with telemetry.stage("embed"): ...  →  kayda t_embed_s alanı ekler."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *exc):
        add(f"t_{self.name}_s", round(time.perf_counter() - self.t, 3))
        return False


def finish_query(**fields) -> None:
    global last_record
    record = getattr(_local, "record", None)
    if record is None:
        return
    record.update(fields)
    record["t_total_s"] = round(time.perf_counter() - _local.t0, 3)
    last_record = record

    try:
        os.makedirs(os.path.dirname(TELEMETRY_PATH), exist_ok=True)
        with open(TELEMETRY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # telemetri hiçbir koşulda cevabı engellemesin
    _local.record = None


def _summary() -> None:
    """Kayıtlı sorguların aşama bazlı özetini basar (python telemetry.py)."""
    if not os.path.exists(TELEMETRY_PATH):
        print(f"Henüz kayıt yok: {TELEMETRY_PATH}")
        return

    with open(TELEMETRY_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        print("Dosya boş.")
        return

    print(f"Toplam sorgu: {len(records)}  (dosya: {TELEMETRY_PATH})\n")
    timing_keys = sorted({k for r in records for k in r if k.startswith("t_")})
    print(f"{'aşama':<16}{'ort':>8}{'min':>8}{'max':>8}{'n':>6}")
    for key in timing_keys:
        values = [r[key] for r in records if key in r]
        print(f"{key:<16}{sum(values)/len(values):>8.2f}{min(values):>8.2f}"
              f"{max(values):>8.2f}{len(values):>6}")

    refused = sum(1 for r in records if r.get("refused"))
    print(f"\nreddetme: {refused}/{len(records)}"
          f"  |  profil dağılımı: { {p: sum(1 for r in records if r.get('profile')==p) for p in {r.get('profile') for r in records}} }")


if __name__ == "__main__":
    _summary()
