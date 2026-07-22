import re
import sys
import time

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

from rag import telemetry
from rag.config import CHAT_MODEL_ALIAS, PARENT_DOC, TOP_K
from rag.retrieval import expand_parents, get_top_chunks

# Bilgi tabanı ve sorular İngilizce olduğu için system prompt ve "bilmiyorum"
# cümlesi de İngilizce — dil tutarlılığı modelin davranışını çok iyileştiriyor.
# Cümleyi sabit tutmak "bilmiyorum" durumunu güvenilir tespit etmemizi sağlıyor.
NO_ANSWER = "I don't have that information."

# Dengeli prompt (2026-07-22): İlk sert sürüm over-refuse ediyordu (cevabı context'ten ÇIKARILABİLEN
# soruları da reddetti — "never guess/refuse instead" aşırı agresifti). Bu sürüm iyi yanları TUTAR
# (grounding + anti-deflection + kod disiplini + directness) ama: (1) çıkarım/sentezi AÇIKÇA izinler,
# (2) reddetmeyi SON ÇARE + dar tetikleyici ("context'te soruyu cevaplayan hiçbir bilgi yoksa") yapar,
# (3) anti-deflection'ı GENEL tutar (codename örneği vermeden — teaching-to-the-test olmasın).
SYSTEM_PROMPT = (
    "You are a knowledgeable technical documentation assistant for Microsoft Foundry Local. "
    "Answer using ONLY the provided CONTEXT - never use outside or pre-trained knowledge.\n\n"
    "- Answer whenever the CONTEXT states the answer OR it can be reasonably concluded by combining "
    "or interpreting what the CONTEXT says (this includes yes/no and multi-part questions). Do not "
    "refuse just because the exact wording is missing.\n"
    "- Every claim you make must be grounded in the CONTEXT; do not add facts from outside knowledge.\n"
    "- When the CONTEXT contains CLI commands, code, or configuration relevant to the answer, include "
    "them in Markdown code blocks. Start directly with the answer or the steps - no filler such as "
    "\"Based on the context\".\n"
    "- Refuse ONLY when the CONTEXT contains no information that answers the question. In that case, "
    "do not guess and do not offer a related or more general term in place of the specific thing "
    f"asked; output ONLY this exact sentence and nothing else: \"{NO_ANSWER}\""
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[Source: {chunk['source']}"
        if chunk.get("heading_path"):
            header += f" | {chunk['heading_path']}"
        header += "]"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def strip_thinking(text: str) -> str:
    """qwen3 gibi reasoning modellerinin ürettiği <think>...</think> bloklarını
    cevaptan çıkarır. Kapalı blok varsa siler; kapanmadan kesilmişse (max_tokens'a
    takılıp </think> gelmemişse) baştaki 'düşünme' kısmını atar. phi-3.5-mini gibi
    reasoning olmayan modellerde <think> hiç üretilmez, bu fonksiyon no-op'tur."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:  # kapanmamış blok (cevap kesilmiş) — kalanı at
        text = text.split("<think>", 1)[0].strip()
    return text


def is_refusal(answer: str) -> bool:
    """Model 'bilmiyorum' dedi mi?

    Hafta 5 turlarında öğrenilen üç ders:
    - Sondaki noktayı sabit arama kaçırıyor (model cümleye devam edebiliyor).
    - Model NO_ANSWER'ı bazen apostrofsuz yazıyor: "I do not have that information".
    - Model bazen kendi kelimeleriyle reddediyor ("The context does not provide...").
      Bu kalıplar sadece cevabın BAŞINDA aranmalı — geçerli cevaplar bazen sonda
      "...the exact comparison is not provided" gibi uyarılarla bitiyor, onları
      reddetme saymak yanlış olur.
    """
    text = answer.lower()
    # Modelin yazım sürçmelerine tolerans: ardışık tekrar eden kelimeleri teke indir
    # ("I don't have that that information" vakası — Foundry eval'inde görüldü)
    text = re.sub(r"\b(\w+)(?: \1\b)+", r"\1", text)

    if "i don't have that information" in text or "i do not have that information" in text:
        return True

    # Model kendi kelimeleriyle de reddedebiliyor: "... is not provided in the given context",
    # "the context does not provide ...", "no information about ... in the documentation". Sabit
    # string yerine regex — "given/provided context", "documentation" gibi varyantları yakalar
    # (sert prompt sonrası model tam NO_ANSWER cümlesini kullanmayıp parafraz ediyordu). SADECE
    # ilk 120 karakterde aranır: geçerli cevaplar sonda "...is not provided in the context" gibi
    # uyarıyla bitebiliyor, onları reddetme saymak yanlış olur.
    head = text[:120]
    refusal_patterns = (
        r"\bnot (?:provided|available|mentioned|found|specified|listed|included|present|contained|documented) in\b.*\b(?:context|documentation)\b",
        r"\b(?:context|documentation)(?: (?:provided|given))? (?:does not|doesn'?t|did not) (?:provide|contain|mention|include|specify|have|list)\b",
        r"\bno (?:information|details?|mention|reference|data)\b.*\b(?:context|documentation)\b",
    )
    return any(re.search(p, head) for p in refusal_patterns)


def answer_query(question: str, chat_client, top_k: int = TOP_K, verbose: bool = False,
                 on_delta=None) -> tuple[str, list[str]]:
    """Retrieval (get_top_chunks) + generation (chat_client) adımlarını birleştirir.

    (cevap_metni, kaynak_listesi) döner. Kaynaklar modelden değil, retrieval'ın
    gerçekten getirdiği chunk'lardan deterministik olarak çıkarılır. Model
    'bilmiyorum' derse kaynak listesi boş döner (yanıltıcı atıf olmasın diye).
    verbose=True ise retrieval'ın getirdiği chunk'ları loglar (PDF Hafta 4:
    "log retrieved chunks for verification").

    on_delta: opsiyonel callback. Verilirse yanıt SDK'nın streaming API'siyle
    üretilir ve her metin parçası geldikçe on_delta(parça) çağrılır — CLI'da
    "yazarak cevaplama" deneyimi sağlar (algılanan gecikmeyi düşürür). Dönüş
    değeri her iki modda da aynıdır; evaluate.py callback vermeden eski
    davranışla çalışmaya devam eder.
    """
    # Edge case: boş/boşluk soru — embedding client boş string'e ValueError
    # fırlatıyor, LLM'e gitmeden kibarca reddet.
    question = question.strip()
    if not question:
        return NO_ANSWER, []

    telemetry.start_query(question)  # embed/search aşamaları retrieval.py içinde ölçülür

    chunks = get_top_chunks(question, top_k=top_k)
    if verbose:
        print(f"\n[retrieval] top_{top_k} chunk:")
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk["content"][:80].replace("\n", " ")
            print(f"  #{i} score={chunk['score']:.4f} source={chunk['source']}  {preview}...")

    # Parent-document retrieval: retrieved chunk'ları ait oldukları tam bölüme genişlet
    # (isabetli arama küçük chunk'ta, cevap üretimi tam bağlamla). Kaynak atıfı yine
    # ORİJİNAL retrieved chunk'lardan çıkar (genişletme sadece LLM'in gördüğü içeriği büyütür).
    if PARENT_DOC:
        with telemetry.stage("parent"):
            context_chunks = expand_parents(chunks)
    else:
        context_chunks = chunks
    context = build_context(context_chunks)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    # qwen3 reasoning modellerinde düşünme modunu kapat (/no_think soft-switch) —
    # bu Q&A görevinde <think> blokları hem çıktıyı kirletiyor hem token bütçesini
    # yiyip asıl cevabı kestiriyor hem de gecikmeyi ~10s'e çıkarıyordu.
    if "qwen3" in CHAT_MODEL_ALIAS.lower():
        user_content += " /no_think"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if on_delta is None:
        with telemetry.stage("llm"):
            response = chat_client.complete_chat(messages)
            answer = response.choices[0].message.content.strip()
    else:
        parts = []
        llm_start = time.perf_counter()
        with telemetry.stage("llm"):
            for chunk in chat_client.complete_streaming_chat(messages):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    if not parts:  # ilk token = prefill süresi (GPU teşhisinin kalıcı ölçümü)
                        telemetry.add("t_first_token_s", round(time.perf_counter() - llm_start, 3))
                    parts.append(delta)
                    on_delta(delta)
        answer = "".join(parts).strip()

    answer = strip_thinking(answer)  # reasoning modeli <think> bloğu bıraktıysa temizle
    refused = is_refusal(answer)
    if refused:
        sources = []
    else:
        # Benzersiz kaynaklar, retrieval sırasını koruyarak (dict.fromkeys sırayı korur)
        sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    telemetry.finish_query(refused=refused, n_sources=len(sources), top_k=top_k,
                           answer_chars=len(answer), streamed=on_delta is not None)
    return answer, sources


def _register_gpu_eps(manager) -> None:
    """Varsa GPU execution provider'ını (CUDA) bu process'e kaydet.

    EP kaydı process başına gerekiyor (kalıcı değil — temiz process'te
    'Available EPs: [CPUExecutionProvider]' hatasıyla öğrenildi). EP binary'leri
    ilk seferde indirilir, sonraki başlangıçlarda kayıt hızlıdır. CUDA
    keşfedilemiyorsa (GPU'suz makine) hiçbir şey yapmaz — CPU'ya düşülür.
    """
    try:
        discovered = {ep.name: ep for ep in manager.discover_eps()}
        cuda = discovered.get("CUDAExecutionProvider")
        if cuda is not None and not cuda.is_registered:
            manager.download_and_register_eps(names=["CUDAExecutionProvider"])
    except FoundryLocalException as exc:
        print(f"[uyari] GPU EP kaydi basarisiz, CPU ile devam: {exc}")


def load_chat_client():
    config = Configuration(app_name="LocalRagAssistant")
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    _register_gpu_eps(manager)

    model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model bulunamadı: {CHAT_MODEL_ALIAS}")

    # GPU varyantı katalogda görünüyorsa (CUDA EP kayıtlıysa) onu tercih et.
    # Not: Model sınıfı cache'lenmiş varyantı otomatik tercih edebiliyor, bu
    # yüzden seçim açıkça yapılıyor. GPU'suz makinede cuda varyantı katalogda
    # hiç görünmez → varsayılan (CPU) seçili kalır.
    gpu_variant = next((v for v in model.variants if "cuda-gpu" in v.id.lower()), None)
    if gpu_variant is not None:
        model.select_variant(gpu_variant)

    if not model.is_cached:
        print(f"Downloading '{CHAT_MODEL_ALIAS}'...")
        model.download(progress_callback=lambda pct: print(f"  %{pct:.1f}", end="\r"))
        print()

    model.load()
    chat_client = model.get_chat_client()
    # Yanıtı sınırla: hem gecikmeyi düşürür hem modelin gereksiz uzun/dağınık
    # cevaplar üretmesini engeller (Q&A için kısa cevap zaten daha iyi).
    chat_client.settings.max_tokens = 256
    # Deterministik çıktı: aynı soru her seferinde aynı cevabı versin. Değerlendirmenin
    # tekrarlanabilir olması ve sınırdaki soruların turdan tura zıplamaması için.
    chat_client.settings.temperature = 0.0
    return chat_client


def main():
    # --verbose bayrağı: retrieval'ın getirdiği chunk'ları göster (demo/hata ayıklama)
    verbose = "--verbose" in sys.argv

    print("Loading model...")
    chat_client = load_chat_client()

    print("Foundry Local Assistant ready. Type 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        # Streaming: token'lar geldikçe ekrana basılır (algılanan gecikme ~2-3s'e iner).
        # "Cevap:" etiketi ilk token'la birlikte basılır ki --verbose'un retrieval
        # logları etiketle cevabın arasına girmesin.
        label_printed = [False]

        def show_delta(text: str) -> None:
            if not label_printed[0]:
                print("\nAnswer: ", end="", flush=True)
                label_printed[0] = True
            print(text, end="", flush=True)

        answer, sources = answer_query(question, chat_client, verbose=verbose, on_delta=show_delta)
        if not label_printed[0]:
            # Hiç token akmadıysa (ör. guard'dan dönen sabit cevap) etiketi yine bas
            print(f"\nAnswer: {answer}", end="")
        print()
        if sources:
            print(f"Sources: {', '.join(sources)}")
        if verbose and telemetry.last_record:
            r = telemetry.last_record
            timings = "  ".join(f"{k.removeprefix('t_').removesuffix('_s')}={v}s"
                                for k, v in r.items() if k.startswith("t_"))
            print(f"[timings] {timings}")
        print()


if __name__ == "__main__":
    main()
