"""telemetry modülü birim testleri — kayıt yaşam döngüsü ve no-op güvenliği."""
import json

from rag import telemetry


def test_stage_without_active_query_is_noop():
    # evaluate.py get_top_chunks'ı bağımsız çağırıyor — aktif kayıt yokken
    # stage() ve add() sessizce geçmeli, hata da kayıt da üretmemeli
    with telemetry.stage("embed"):
        pass
    telemetry.add("orphan", 1)  # patlamamalı


def test_finish_without_start_writes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "t.jsonl"
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", str(out))
    telemetry.finish_query(refused=False)
    assert not out.exists()


def test_full_query_lifecycle_writes_one_record(tmp_path, monkeypatch):
    out = tmp_path / "t.jsonl"
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", str(out))

    telemetry.start_query("What is Foundry Local?")
    with telemetry.stage("embed"):
        pass
    telemetry.add("t_first_token_s", 0.5)
    telemetry.finish_query(refused=False, n_sources=2, top_k=3)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["question_chars"] == len("What is Foundry Local?")
    assert record["n_sources"] == 2
    assert record["refused"] is False
    assert "t_embed_s" in record
    assert record["t_first_token_s"] == 0.5
    assert "t_total_s" in record
    # last_record CLI/UI için erişilebilir olmalı
    assert telemetry.last_record == record


def test_two_queries_append_two_lines(tmp_path, monkeypatch):
    out = tmp_path / "t.jsonl"
    monkeypatch.setattr(telemetry, "TELEMETRY_PATH", str(out))
    for _ in range(2):
        telemetry.start_query("q")
        telemetry.finish_query(refused=True, n_sources=0, top_k=3)
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2
