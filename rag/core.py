import re
import sys
import time

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

from rag import telemetry
from rag.config import CHAT_MODEL_ALIAS, CONDENSE_HISTORY, HISTORY_TURNS, PARENT_DOC, TOP_K
from rag.retrieval import expand_parents, get_top_chunks

# Bilgi tabanı İngilizce, o yüzden prompt ve "bilmiyorum" cümlesi de İngilizce; dil tutarlılığı
# modelin cevaplarını belirgin iyileştiriyor. Cümleyi sabit tutmak reddetmeyi güvenilir tespit ettiriyor.
NO_ANSWER = "I don't have that information."

# Grounded cevap prompt'u. İki uçta da sorun yaşamıştık: kural yokken model alakasız kurulum
# adımlarını döküyor, "kısa tut" deyince de tek kelimelik cevap veriyordu. Şu anki denge: düzgün
# cevapla ama uzunluğu soruya göre ayarla, alakasızı ekleme; reddetmeyi son çare olarak tut.
SYSTEM_PROMPT = (
    "You are a knowledgeable technical documentation assistant for Microsoft Foundry Local. "
    "Answer using ONLY the provided CONTEXT - never use outside or pre-trained knowledge.\n\n"
    "- Answer whenever the CONTEXT states the answer OR it can be reasonably concluded by combining "
    "or interpreting what the CONTEXT says (this includes yes/no and multi-part questions). Do not "
    "refuse just because the exact wording is missing.\n"
    "- Answer properly using the relevant details from the CONTEXT: do not give a bare one-line "
    "answer when the context supports a fuller one, and do not dump unrelated setup steps or "
    "tangential sections. Match the length to the question - a factual question needs a sentence or "
    "two of explanation; a how-to needs its steps. Do not pad beyond what the question needs.\n"
    "- Every claim you make must be grounded in the CONTEXT; do not add facts from outside knowledge. "
    "Only reproduce commands, code, URLs, and file paths that appear in the CONTEXT VERBATIM - NEVER "
    "invent, guess, or construct a command, URL, or path that is not in the CONTEXT (if the CONTEXT "
    "only gives a download link or a description, give that, do not fabricate a command for it). When "
    "the CONTEXT has a CLI command or code that DIRECTLY answers the question, include it in a Markdown "
    "code block. Start directly with the answer - no filler such as \"Based on the context\".\n"
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
    """qwen3 gibi reasoning modellerinin ürettiği <think>...</think> bloklarını cevaptan siler.
    Kapalı blok tamamen çıkarılır; token limitine takılıp kapanmamışsa baştaki düşünme kısmı atılır.
    phi-3.5-mini gibi modeller <think> üretmez, o durumda bir şey yapmaz."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:  # blok kapanmamış (cevap kesilmiş), kalanı at
        text = text.split("<think>", 1)[0].strip()
    return text


def is_refusal(answer: str) -> bool:
    """Model 'bilmiyorum' mu dedi?

    Birkaç incelik var:
    - Model cümleye devam edebildiği için sondaki noktaya güvenilmez.
    - "I do not have that information" gibi apostrofsuz varyantlar da geçebiliyor.
    - Model bazen kendi kelimeleriyle reddediyor ("The context does not provide..."); bu kalıpları
      sadece cevabın başında ararız, çünkü geçerli cevaplar da sonda benzer bir uyarıyla bitebiliyor.
    """
    text = answer.lower()
    # Yazım sürçmelerine tolerans: ardışık tekrar eden kelimeleri teke indir
    # ("I don't have that that information" gibi)
    text = re.sub(r"\b(\w+)(?: \1\b)+", r"\1", text)

    if "i don't have that information" in text or "i do not have that information" in text:
        return True

    # Model reddetmeyi parafraz edebiliyor ("... is not provided in the given context", "the context
    # does not provide ..."). Sabit string yerine regex kullanıyoruz ve sadece ilk 120 karakterde
    # arıyoruz, çünkü geçerli cevaplar da sonda benzer bir uyarı cümlesiyle bitebiliyor.
    head = text[:120]
    refusal_patterns = (
        r"\bnot (?:provided|available|mentioned|found|specified|listed|included|present|contained|documented) in\b.*\b(?:context|documentation)\b",
        r"\b(?:context|documentation)(?: (?:provided|given))? (?:does not|doesn'?t|did not) (?:provide|contain|mention|include|specify|have|list)\b",
        r"\bno (?:information|details?|mention|reference|data)\b.*\b(?:context|documentation)\b",
    )
    return any(re.search(p, head) for p in refusal_patterns)


# Takip sorusunu bağımsızlaştırma prompt'u. İlk sürüm "make it self-contained" deyince model önceki
# turun konusunu gereksizce ekliyordu ("Does it work with LangChain?" -> "Does tool calling ... work
# with LangChain?"), yanlış doküman geliyordu. Şimdi sadece zamiri çözüyor, başka bir şey eklemiyor;
# örnekler zamiri hem ürüne hem eyleme çözmeyi gösteriyor.
CONDENSE_SYSTEM_PROMPT = (
    "Rewrite the user's follow-up question into a standalone question by replacing each referring "
    "expression (it, that, those, they, the previous one, etc.) with the SPECIFIC thing it refers "
    "to in the conversation - which may be an object/product OR a task/action, whichever the word "
    "actually points to. Change as FEW words as possible. Do NOT add any topic, scope, clause, or "
    "comparison that the follow-up did not explicitly ask about. If the follow-up is already "
    "standalone, output it unchanged. Output ONLY the rewritten question on one line - no "
    "explanation, no quotes.\n\n"
    "Example 1 (reference points to a product):\n"
    "Conversation: the user asked how to use tool calling with Foundry Local.\n"
    "Follow-up: Does it work with LangChain?\n"
    "Rewritten: Does Foundry Local work with LangChain?\n\n"
    "Example 2 (reference points to a task):\n"
    "Conversation: the user asked how to generate embeddings with Foundry Local.\n"
    "Follow-up: Can I do that in JavaScript?\n"
    "Rewritten: Can I generate embeddings with Foundry Local in JavaScript?"
)

# Coreference/ellipsis işaretleri — takip sorusunun condensation'a ihtiyacı olup olmadığını LLM'siz
# belirler. Kendi içinde bağımsız yeni sorular condense edilmez; yoksa model önceki turun konusunu
# sürükleyip retrieval'ı yanlış dokümana çekiyor.
_ANAPHORA = re.compile(
    r"\b(it|its|it's|that|this|these|those|them|they|their|theirs|one|ones|same|"
    r"above|former|latter|previous)\b", re.IGNORECASE)
_FOLLOWUP_START = re.compile(
    r"^\s*(and|but|or|so|also|then|what about|how about|what if)\b", re.IGNORECASE)


def needs_condensation(question: str) -> bool:
    """Takip sorusu bağlama bağımlı mı (coreference/ellipsis)? Değilse condensation atlanır — hem
    konu sürüklemesini hem gereksiz bir LLM çağrısını önler. Çok kısa sorular (<=3 kelime) genelde
    eliptik olduğu için ("In Rust?", "On GPU?") yine de condense edilir."""
    q = question.strip()
    if _ANAPHORA.search(q) or _FOLLOWUP_START.search(q):
        return True
    return len(q.split()) <= 3


def condense_query(question: str, history, chat_client) -> str:
    """Takip sorusunu retrieval için bağımsız sorguya çevirir; zamiri konuşmadan çözer. Korpus
    terimi değil konuşmadaki bilgiyi kullandığı için query-expansion'dan farklı. Sonuç boş ya da
    aşırı uzunsa orijinal soruya döner."""
    convo = "\n".join(f"User: {t['question']}\nAssistant: {t['answer'][:200]}"
                      for t in history[-HISTORY_TURNS:])
    user_content = f"Conversation:\n{convo}\n\nFollow-up question: {question}"
    if "qwen3" in CHAT_MODEL_ALIAS.lower():
        user_content += " /no_think"
    response = chat_client.complete_chat([
        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    rewritten = strip_thinking(response.choices[0].message.content.strip())
    rewritten = rewritten.splitlines()[0].strip().strip('"') if rewritten else ""
    return rewritten if rewritten and len(rewritten) <= 300 else question


def answer_query(question: str, chat_client, top_k: int = TOP_K, history=None,
                 verbose: bool = False, on_delta=None, on_retrieval=None) -> tuple[str, list[str]]:
    """Retrieval ve generation adımlarını birleştirir.

    (cevap, kaynak_listesi) döner. Kaynaklar modelden değil, retrieval'ın gerçekten getirdiği
    chunk'lardan çıkarılır; model reddederse liste boş döner ki yanıltıcı atıf olmasın.
    verbose=True getirilen chunk'ları loglar.

    history: opsiyonel önceki turlar. Verilirse sadece retrieval'ı düzeltmek için kullanılır —
    takip sorusunu condense_query ile bağımsızlaştırır (son HISTORY_TURNS tur). Generation tek-tur
    kalır, geçmiş prompt'a girmez. None verilince tek-turlu davranır (eval böyle çağırıyor).

    on_delta: opsiyonel callback. Verilirse cevap streaming ile üretilir ve her parça geldikçe
    çağrılır (yazarak cevaplama deneyimi). Dönüş değeri iki modda da aynı.

    on_retrieval: opsiyonel callback(search_query, chunks). Retrieval biter bitmez modele giden
    gerçek sorgu ve chunk'larla çağrılır. UI'ın "modelin gördüğü chunk'lar" panelini ikinci bir
    arama yapmadan doldurması için; multi-turn'de bu sorgu condense edilmiş halidir.
    """
    # Boş soru: embedding client boş string'e hata veriyor, LLM'e gitmeden kibarca reddet.
    question = question.strip()
    if not question:
        return NO_ANSWER, []

    telemetry.start_query(question)  # embed/search aşamaları retrieval.py içinde ölçülür

    # Takip sorusunu geçmişle bağımsız sorguya çevirip retrieval'ı düzeltiyoruz; cevap yine orijinal
    # soruyla üretilir. Sadece bağlama bağımlı takip sorularında condense ediyoruz — bağımsız yeni
    # sorular ("What is Foundry Local?") ham haliyle aranıyor, böylece eski konu sürüklenmiyor ve
    # gereksiz bir LLM çağrısı olmuyor. history yoksa (tek-tur/eval) ham soru kullanılır.
    search_query = question
    if history and CONDENSE_HISTORY and needs_condensation(question):
        with telemetry.stage("condense"):
            search_query = condense_query(question, history, chat_client)
        if verbose and search_query != question:
            print(f"[condense] '{question}' -> '{search_query}'")

    chunks = get_top_chunks(search_query, top_k=top_k)
    if on_retrieval is not None:  # UI şeffaflık paneli için: modele giden gerçek sorgu ve chunk'lar
        on_retrieval(search_query, chunks)
    if verbose:
        print(f"\n[retrieval] top_{top_k} chunk:")
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk["content"][:80].replace("\n", " ")
            print(f"  #{i} score={chunk['score']:.4f} source={chunk['source']}  {preview}...")

    # Retrieved chunk'ları ait oldukları tam dokümana genişlet: arama küçük chunk'ta isabetli,
    # cevap üretimi tam bağlamla. Kaynak atıfı yine orijinal chunk'lardan çıkar.
    if PARENT_DOC:
        with telemetry.stage("parent"):
            context_chunks = expand_parents(chunks)
    else:
        context_chunks = chunks
    context = build_context(context_chunks)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    # qwen3'te düşünme modunu kapat: bu görevde <think> blokları çıktıyı kirletiyor, token
    # bütçesini yiyor ve gecikmeyi artırıyordu.
    if "qwen3" in CHAT_MODEL_ALIAS.lower():
        user_content += " /no_think"
    # Condense-only mimari: geçmiş sadece retrieval'a gitti, generation tek-tur. Model önceki cevabını
    # görmediği için grounding taze bağlamdan geliyor. Önceki cevapları prompt'a koymayı denedik ama
    # yanlış atıflara yol açtı, kaldırdık.
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
                    if not parts:  # ilk token = prefill süresi (gecikme teşhisi için ölçülür)
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
    """Varsa CUDA execution provider'ını bu process'e kaydeder.

    EP kaydı her process'te tekrar gerekiyor. Binary'ler ilk seferde iniyor, sonraki
    başlangıçlar hızlı. GPU yoksa hiçbir şey yapmaz, CPU'ya düşülür.
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

    # GPU varyantı katalogda görünüyorsa onu seç. Seçimi açıkça yapıyoruz çünkü model sınıfı
    # bazen cache'lenmiş varyantı tercih ediyor. GPU'suz makinede cuda varyantı görünmez, CPU kalır.
    gpu_variant = next((v for v in model.variants if "cuda-gpu" in v.id.lower()), None)
    if gpu_variant is not None:
        model.select_variant(gpu_variant)

    if not model.is_cached:
        print(f"Downloading '{CHAT_MODEL_ALIAS}'...")
        model.download(progress_callback=lambda pct: print(f"  %{pct:.1f}", end="\r"))
        print()

    model.load()
    chat_client = model.get_chat_client()
    # Yanıt token bütçesi. 256 düşüktü, kod bloklu/adım adım cevaplar cümle ortasında kesiliyordu;
    # 512 yeterli pay bırakıyor. Dağınıklığı token limiti değil, prompt'taki odak kuralı dizginliyor.
    chat_client.settings.max_tokens = 512
    # Deterministik çıktı: aynı soru hep aynı cevabı versin, eval tekrarlanabilir olsun.
    chat_client.settings.temperature = 0.0
    return chat_client


def main():
    # --verbose: getirilen chunk'ları da göster (hata ayıklama)
    verbose = "--verbose" in sys.argv

    print("Loading model...")
    chat_client = load_chat_client()

    print("Foundry Local Assistant ready. Type 'exit' to quit.\n")
    history: list[dict] = []  # önceki soru-cevaplar; condensation son HISTORY_TURNS'ü kullanır
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        # Streaming: token'lar geldikçe ekrana basılıyor. "Answer:" etiketini ilk token'la
        # basıyoruz ki --verbose'un retrieval logları etiketle cevabın arasına girmesin.
        label_printed = [False]

        def show_delta(text: str) -> None:
            if not label_printed[0]:
                print("\nAnswer: ", end="", flush=True)
                label_printed[0] = True
            print(text, end="", flush=True)

        answer, sources = answer_query(question, chat_client, history=history,
                                        verbose=verbose, on_delta=show_delta)
        history.append({"question": question, "answer": answer})
        if not label_printed[0]:
            # Hiç token akmadıysa (ör. boş soru guard'ı) etiketi yine bas
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
