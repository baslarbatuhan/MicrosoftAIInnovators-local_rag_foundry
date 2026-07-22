"""is_refusal() birim testleri — Hafta 5'ten beri biriken tüm reddetme
varyantları ve yanlış-pozitif tuzakları testle sabitleniyor."""
from rag.core import NO_ANSWER, build_context, is_refusal


# --- Reddetme sayılması gerekenler (hepsi gerçek eval turlarında görüldü) ---

def test_exact_no_answer_sentence():
    assert is_refusal(NO_ANSWER)


def test_no_answer_with_prefix_and_suffix():
    assert is_refusal("Sadece: I don't have that information.")
    assert is_refusal("I don't have that information based on the provided context.")


def test_apostrophe_free_variant():
    assert is_refusal("Therefore, I do not have that information.")


def test_word_stutter_variant():
    # Foundry eval'inde görüldü: model "that that" yazım sürçmesi yapabiliyor
    assert is_refusal("I don't have that that information.")


def test_context_does_not_provide_at_head():
    assert is_refusal("The context does not provide a specific number for the health.")
    assert is_refusal("The provided context does not contain pricing details.")
    assert is_refusal("The internal codename is not provided in the context.")


def test_paraphrased_refusal_given_context():
    # Sert prompt sonrası model tam NO_ANSWER cümlesini kullanmayıp parafraz ediyor:
    # "... is not provided in the GIVEN context" (eski marker 'given' yüzünden kaçırıyordu).
    assert is_refusal(
        "The internal Microsoft codename for the Foundry Local project is not provided "
        "in the given context. To find this information, refer to internal documentation."
    )
    assert is_refusal("This detail is not mentioned in the provided documentation.")


# --- Reddetme SAYILMAMASI gerekenler ---

def test_normal_answer_is_not_refusal():
    assert not is_refusal("Keybullet Kin drop a key upon death.")
    assert not is_refusal("Run winget install Microsoft.FoundryLocal in a terminal.")


def test_trailing_caveat_after_valid_answer_is_not_refusal():
    # Değerlendirmede görüldü: geçerli cevap sonda uyarı cümlesiyle bitebiliyor.
    # Kalıplar sadece cevabın BAŞINDA (ilk 120 karakter) aranmalı.
    answer = (
        "The supported text index databases are MongoDB, Postgres and SQLite, while the "
        "supported vector databases include Milvus, Redis and FAISS for similarity search. "
        "However, the exact performance comparison is not provided in the context."
    )
    assert not is_refusal(answer)


# --- build_context format sözleşmesi ---

def test_build_context_tags_sources():
    chunks = [
        {"source": "a.txt", "content": "Alpha content", "score": 0.9},
        {"source": "b.txt", "content": "Beta content", "score": 0.8},
    ]
    context = build_context(chunks)
    assert "[Source: a.txt]\nAlpha content" in context
    assert "[Source: b.txt]\nBeta content" in context
    assert context.count("---") == 1  # chunk ayracı
