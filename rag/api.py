"""
FastAPI servis katmanı — RAG motorunu (answer_query) HTTP üzerinden sunar.

Motoru sarmalıyoruz, pipeline mantığını tekrarlamıyoruz. Model bir kez lifespan'de yüklenip
app.state'te tutuluyor (Streamlit'teki @cache_resource'un karşılığı). Motor stateless olduğu
için API de stateless: client geçmişi gönderiyor, server durum tutmuyor.

Tek yüklü yerel model aynı anda tek inference yapabildiği için answer_query çağrısını bir
asyncio.Lock ile serileştiriyoruz (kişisel/düşük trafikli kullanım için yeterli; çok kullanıcı
için model havuzu gerekir). Bloklayıcı motoru threadpool'da koşturuyoruz ki event loop donmasın.

Çalıştırma:  uvicorn rag.api:app       (veya: python -m rag.api)
Varsayılan 127.0.0.1:8000, dışa açık değil (auth/rate-limit yok).
"""
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from rag.config import TOP_K
from rag.core import answer_query, load_chat_client
from rag.retrieval import get_top_chunks


# --- Şema (Pydantic) ---

class Turn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="Kullanıcının sorusu (İngilizce).")
    history: list[Turn] = Field(
        default_factory=list, description="Önceki turlar — stateless: client tutar ve gönderir."
    )
    top_k: int = Field(default=TOP_K, ge=1, le=10)
    include_retrieval: bool = Field(
        default=False, description="True ise condense'lenmiş sorgu + getirilen chunk'lar da döner."
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("question boş olamaz")
        return v


class ChunkInfo(BaseModel):
    source: str
    heading_path: str = ""
    score: float | None = None
    content: str


class RetrievalInfo(BaseModel):
    search_query: str  # multi-turn'de condense edilmiş sorgu; ham sorudan farklıysa coreference çözülmüştür
    chunks: list[ChunkInfo]


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval: RetrievalInfo | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Retrieval sorgusu (ham; condensation uygulanmaz).")
    top_k: int = Field(default=TOP_K, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query boş olamaz")
        return v


class SearchResponse(BaseModel):
    chunks: list[ChunkInfo]


# --- Servis (motoru sarmalar) ---

class RagService:
    """Model client'ını tutar ve inference'ı serileştirir. answer(), bloklayıcı answer_query'yi
    kilit altında threadpool'da koşturur ki event loop donmasın ve modele aynı anda tek istek gitsin."""

    def __init__(self, chat_client):
        self.chat_client = chat_client
        self._lock = asyncio.Lock()

    async def answer(self, question: str, history: list[dict], top_k: int, include_retrieval: bool):
        captured: dict = {}

        def on_retrieval(search_query: str, chunks: list) -> None:
            captured["search_query"] = search_query
            captured["chunks"] = chunks

        loop = asyncio.get_running_loop()
        async with self._lock:  # aynı anda tek inference; telemetry.last_record de böylece güvende
            answer, sources = await loop.run_in_executor(
                None,
                lambda: answer_query(
                    question, self.chat_client, top_k=top_k, history=history,
                    on_retrieval=on_retrieval if include_retrieval else None,
                ),
            )
        return answer, sources, captured

    async def answer_stream(self, question: str, history: list[dict], top_k: int, include_retrieval: bool):
        """Streaming: motorun on_delta callback'ini bir asyncio.Queue'ya bağlar (worker thread
        token'ları koyar, bu async generator alır), böylece motoru değiştirmeden SSE üretiyoruz.
        ("token", metin), sonra ("final", (answer, sources, captured)) ya da ("error", mesaj) döner.
        Kilit generator bitene kadar tutulur; client akış ortasında koparsa worker kendi başına
        tamamlanır (yerel tek kullanıcı için sorun değil)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        captured: dict = {}

        def on_delta(text: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, ("token", text))

        def on_retrieval(search_query: str, chunks: list) -> None:
            captured["search_query"] = search_query
            captured["chunks"] = chunks

        def run() -> None:
            try:
                answer, sources = answer_query(
                    question, self.chat_client, top_k=top_k, history=history,
                    on_delta=on_delta, on_retrieval=on_retrieval if include_retrieval else None,
                )
                loop.call_soon_threadsafe(queue.put_nowait, ("final", (answer, sources, dict(captured))))
            except Exception as exc:  # noqa: BLE001 — hata SSE ile client'a iletilir
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        async with self._lock:  # aynı anda tek inference
            loop.run_in_executor(None, run)  # worker'ı başlat; sonuç queue üzerinden gelecek
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item

    async def search(self, query: str, top_k: int) -> list[dict]:
        """Debug/eval: ham sorgu için retrieval'ın döndürdüğü top_k chunk (condensation ya da LLM yok)."""
        loop = asyncio.get_running_loop()
        async with self._lock:  # embedding/rerank de Foundry'ye gidiyor, aynı kilitle serileştir
            return await loop.run_in_executor(None, lambda: get_top_chunks(query, top_k=top_k))


# --- Uygulama ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model bir kez yükleniyor. Foundry servisi kapalıysa startup'ı çökertmiyoruz; service=None
    # bırakıyoruz, /chat 503 dönüyor ve /health model_loaded=False raporluyor.
    try:
        app.state.service = RagService(load_chat_client())
    except Exception as exc:  # noqa: BLE001 — yükleme hatası startup'ı düşürmesin
        app.state.service = None
        app.state.load_error = str(exc)
        print(f"[api] model yüklenemedi, /chat 503 dönecek: {exc}")
    yield


app = FastAPI(title="Foundry Local RAG Assistant", version="1.0", lifespan=lifespan)


def get_service(request: Request) -> "RagService | None":
    """Yüklü servisi döner. Testler dependency_overrides ile sahte servis geçebilir, böylece
    HTTP sözleşmesi model ya da DB yüklemeden test edilebilir."""
    return getattr(request.app.state, "service", None)


@app.get("/")
def root() -> dict:
    """Kök URL bilgi verir (tarayıcıdan girilince 404 yerine yönlendirme). İnteraktif
    deneme için /docs (Swagger UI); /chat POST'tur, tarayıcı GET'iyle çağrılamaz."""
    return {
        "name": "Foundry Local RAG Assistant",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "chat_stream": "POST /chat/stream (SSE)",
            "search": "POST /search",
        },
    }


@app.get("/health")
def health(service: "RagService | None" = Depends(get_service)) -> dict:
    return {"status": "ok", "model_loaded": service is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, service: "RagService | None" = Depends(get_service)) -> ChatResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil (Foundry servisi çalışıyor mu?).")

    history = [{"question": t.question, "answer": t.answer} for t in req.history]
    answer, sources, captured = await service.answer(
        req.question, history, req.top_k, req.include_retrieval
    )

    retrieval = None
    if req.include_retrieval and "chunks" in captured:
        retrieval = RetrievalInfo(
            search_query=captured["search_query"],
            chunks=[
                ChunkInfo(source=c["source"], heading_path=c.get("heading_path", ""),
                          score=c.get("score"), content=c["content"])
                for c in captured["chunks"]
            ],
        )
    return ChatResponse(answer=answer, sources=sources, retrieval=retrieval)


def _sse(event: str, data: dict) -> str:
    """Server-Sent Events çerçevesi: 'event: <tip>\\ndata: <json>\\n\\n'."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _retrieval_dict(captured: dict) -> dict:
    return {
        "search_query": captured["search_query"],
        "chunks": [
            {"source": c["source"], "heading_path": c.get("heading_path", ""),
             "score": c.get("score"), "content": c["content"]}
            for c in captured["chunks"]
        ],
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, service: "RagService | None" = Depends(get_service)):
    """Token akışını SSE olarak yayınlar: her token bir `event: token`, bitişte `event: done`
    (answer + sources + opsiyonel retrieval), hata olursa `event: error`."""
    if service is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil (Foundry servisi çalışıyor mu?).")
    history = [{"question": t.question, "answer": t.answer} for t in req.history]

    async def event_gen():
        async for kind, payload in service.answer_stream(
            req.question, history, req.top_k, req.include_retrieval
        ):
            if kind == "token":
                yield _sse("token", {"text": payload})
            elif kind == "final":
                answer, sources, captured = payload
                data = {"answer": answer, "sources": sources}
                if req.include_retrieval and "chunks" in captured:
                    data["retrieval"] = _retrieval_dict(captured)
                yield _sse("done", data)
            elif kind == "error":
                yield _sse("error", {"detail": payload})

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, service: "RagService | None" = Depends(get_service)) -> SearchResponse:
    """Debug/eval: ham sorgu için retrieval'ın döndürdüğü chunk'lar (LLM çağrısı yok)."""
    if service is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil (Foundry servisi çalışıyor mu?).")
    chunks = await service.search(req.query, req.top_k)
    return SearchResponse(chunks=[
        ChunkInfo(source=c["source"], heading_path=c.get("heading_path", ""),
                  score=c.get("score"), content=c["content"])
        for c in chunks
    ])


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
