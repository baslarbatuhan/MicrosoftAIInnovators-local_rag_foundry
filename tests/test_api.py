"""rag.api HTTP sözleşme testleri — sahte RagService ile (gerçek model/DB YÜKLENMEZ).

TestClient context-manager OLMADAN kullanılır → lifespan (ağır model yükleme) tetiklenmez;
endpoint'ler dependency_overrides ile sahte servisi kullanır. Böylece request validasyonu,
response şekli ve hata yolları Foundry servisi olmadan, hızlı ve deterministik test edilir.
"""
import pytest
from fastapi.testclient import TestClient

from rag.api import app, get_service


class FakeService:
    """RagService.answer() imzasını taklit eder — sabit veri döner, çağrıları kaydeder."""

    def __init__(self):
        self.calls = []

    def _captured(self, question, include_retrieval):
        if not include_retrieval:
            return {}
        return {
            "search_query": f"[condensed] {question}",
            "chunks": [{"source": "doc.txt", "heading_path": "H > Sub",
                        "score": 0.91, "content": "chunk body"}],
        }

    async def answer(self, question, history, top_k, include_retrieval):
        self.calls.append(
            {"question": question, "history": history, "top_k": top_k,
             "include_retrieval": include_retrieval}
        )
        return "Test answer.", ["doc.txt"], self._captured(question, include_retrieval)

    async def answer_stream(self, question, history, top_k, include_retrieval):
        self.calls.append({"stream": True, "question": question, "include_retrieval": include_retrieval})
        for tok in ("Test ", "answer."):
            yield ("token", tok)
        yield ("final", ("Test answer.", ["doc.txt"], self._captured(question, include_retrieval)))

    async def search(self, query, top_k):
        self.calls.append({"search": True, "query": query, "top_k": top_k})
        return [{"source": "doc.txt", "heading_path": "H", "score": 0.88, "content": "chunk body"}]


@pytest.fixture
def fake():
    svc = FakeService()
    app.dependency_overrides[get_service] = lambda: svc
    yield svc
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)  # context-manager DEĞİL → lifespan/model yükleme yok


def test_root_returns_info(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["docs"] == "/docs"
    assert "chat" in body["endpoints"]


def test_health_reports_loaded(fake, client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_chat_returns_answer_and_sources(fake, client):
    r = client.post("/chat", json={"question": "What is Foundry Local?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Test answer."
    assert body["sources"] == ["doc.txt"]
    assert body["retrieval"] is None  # include_retrieval verilmedi


def test_chat_include_retrieval_returns_chunks(fake, client):
    r = client.post("/chat", json={"question": "q", "include_retrieval": True})
    assert r.status_code == 200
    ret = r.json()["retrieval"]
    assert ret["search_query"] == "[condensed] q"
    assert ret["chunks"][0]["source"] == "doc.txt"
    assert ret["chunks"][0]["score"] == 0.91


def test_chat_passes_history_as_dicts(fake, client):
    r = client.post("/chat", json={
        "question": "Can I do that in JavaScript?",
        "history": [{"question": "How do I generate embeddings?", "answer": "steps..."}],
    })
    assert r.status_code == 200
    # Pydantic Turn'ler motor için ham dict'e çevrilmeli
    assert fake.calls[0]["history"] == [
        {"question": "How do I generate embeddings?", "answer": "steps..."}
    ]


def test_blank_question_rejected(fake, client):
    r = client.post("/chat", json={"question": "   "})
    assert r.status_code == 422


def test_top_k_out_of_range_rejected(fake, client):
    r = client.post("/chat", json={"question": "q", "top_k": 99})
    assert r.status_code == 422


def test_service_unavailable_returns_503(client):
    app.dependency_overrides[get_service] = lambda: None  # model yüklenememiş durumu
    try:
        r = client.post("/chat", json={"question": "q"})
        assert r.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_chat_stream_emits_tokens_and_done(fake, client):
    r = client.post("/chat/stream", json={"question": "q", "include_retrieval": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    text = r.text
    assert "event: token" in text
    assert "event: done" in text
    assert "Test answer." in text          # done event'te tam cevap
    assert "[condensed] q" in text          # done event'te retrieval


def test_search_returns_chunks(fake, client):
    r = client.post("/search", json={"query": "embeddings", "top_k": 2})
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert chunks[0]["source"] == "doc.txt"
    assert chunks[0]["score"] == 0.88
    assert fake.calls[0] == {"search": True, "query": "embeddings", "top_k": 2}


def test_search_blank_query_rejected(fake, client):
    r = client.post("/search", json={"query": "  "})
    assert r.status_code == 422


def test_root_lists_new_endpoints(client):
    r = client.get("/")
    eps = r.json()["endpoints"]
    assert "chat_stream" in eps and "search" in eps
