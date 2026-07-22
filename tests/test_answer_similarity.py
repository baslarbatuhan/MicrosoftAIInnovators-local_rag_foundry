"""split_sentences() birim testleri — max-sim recall metriğinin cümle bölme davranışını sabitler.
(answer_similarity kendisi embedding modeli gerektirdiği için burada test edilmez; split_sentences
deterministik ve modelsizdir.)"""
from rag.evaluate import split_sentences


def test_plain_sentences():
    assert split_sentences("First fact. Second fact. Third fact.") == [
        "First fact.", "Second fact.", "Third fact."]


def test_code_block_is_atomic():
    # Kod bloğu içindeki satır sonları/noktalar cümleye BÖLÜNMEZ — tek "iddia" olarak kalır.
    text = "Install it.\n```bash\npip install foundry-local-sdk\ncd app\n```\nThen run it."
    sents = split_sentences(text)
    assert "```bash\npip install foundry-local-sdk\ncd app\n```" in sents
    assert "Install it." in sents
    assert "Then run it." in sents


def test_short_fragments_dropped():
    # 3 karakterden kısa parçalar elenir (gürültü)
    assert split_sentences("OK. This is a real sentence.") == ["This is a real sentence."]


def test_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
