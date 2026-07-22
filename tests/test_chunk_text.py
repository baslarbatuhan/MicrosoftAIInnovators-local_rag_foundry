"""chunk_text() birim testleri — paragraf birleştirme, sınır ve overlap davranışı.

Bu fonksiyon Hafta 5'te en çok ders çıkarılan yerdi (sınır kopmaları, overlap);
davranışı artık testle sabitleniyor ki gelecekteki değişiklikler regresyonu
CI'da yakalasın.
"""
from rag.ingest import chunk_markdown, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("\n\n\n") == []


def test_single_short_paragraph_is_one_chunk():
    assert chunk_text("Hello world.") == ["Hello world."]


def test_consecutive_paragraphs_merge_under_limit():
    text = "First paragraph.\n\nSecond paragraph."
    assert chunk_text(text, max_chars=200) == ["First paragraph.\n\nSecond paragraph."]


def test_splits_when_limit_exceeded_overlap_skipped_when_too_big():
    # 500+2+500 > 800: overlap taşınamaz (taşınsaydı chunk sınırı aşardı) —
    # bu durumda yeni chunk overlap'sız başlar. Sığan durumdaki overlap davranışı
    # bir sonraki testte doğrulanıyor.
    para_a = "A" * 500
    para_b = "B" * 500
    chunks = chunk_text(f"{para_a}\n\n{para_b}", max_chars=800)
    assert chunks == [para_a, para_b]


def test_overlap_carries_last_paragraph_of_previous_chunk():
    paras = ["A" * 300, "B" * 300, "C" * 300]
    chunks = chunk_text("\n\n".join(paras), max_chars=700)
    # A+B sığar (603), C ile 904 > 700 → böl; yeni chunk B ile başlamalı
    assert len(chunks) == 2
    assert chunks[0] == f"{paras[0]}\n\n{paras[1]}"
    assert chunks[1] == f"{paras[1]}\n\n{paras[2]}"


def test_oversized_paragraph_stands_alone_without_overlap():
    huge = "X" * 900          # tek başına max_chars'ı aşıyor
    small = "y" * 100
    chunks = chunk_text(f"{small}\n\n{huge}", max_chars=800)
    # huge kendi chunk'ında; small+huge birleşmez, huge sonrasına overlap taşınmaz
    assert chunks[0] == small
    assert chunks[1] == huge


def test_no_chunk_exceeds_limit_except_oversized_paragraphs():
    paras = [f"Paragraph number {i} " + "word " * 30 for i in range(20)]
    for chunk in chunk_text("\n\n".join(paras), max_chars=800):
        assert len(chunk) <= 800


# --- chunk_markdown: başlık-farkındalıklı bölme + heading_path ---

def test_chunk_markdown_builds_heading_path():
    md = (
        "# Install Foundry Local\n\n"
        "Intro text.\n\n"
        "## Windows\n\n"
        "Run winget install.\n\n"
        "## Linux\n\n"
        "Use the shell script."
    )
    chunks = chunk_markdown(md, max_chars=800)
    paths = {c["heading_path"] for c in chunks}
    assert "Install Foundry Local" in paths
    assert "Install Foundry Local > Windows" in paths
    assert "Install Foundry Local > Linux" in paths
    # içerik başlık satırını değil gövdeyi taşımalı
    win = next(c for c in chunks if c["heading_path"].endswith("Windows"))
    assert "Run winget install." in win["content"]
    assert "#" not in win["content"]


def test_chunk_markdown_resets_deeper_levels():
    md = "# A\n\n## B\n\n### C\n\ncontent c\n\n## D\n\ncontent d"
    chunks = chunk_markdown(md, max_chars=800)
    d = next(c for c in chunks if c["content"].strip() == "content d")
    # D, B'nin kardeşi — path 'A > D' olmalı, 'C' düşmeli
    assert d["heading_path"] == "A > D"


def test_chunk_markdown_cleans_tab_links_in_headings():
    md = "## [Windows](#tab/windows)\n\nsome content"
    chunks = chunk_markdown(md, max_chars=800)
    assert chunks[0]["heading_path"] == "Windows"


def test_chunk_markdown_splits_long_sections_keeping_path():
    long_body = "\n\n".join("X" * 400 for _ in range(4))  # ~1600 char, tek bölüm
    md = f"# Big\n\n{long_body}"
    chunks = chunk_markdown(md, max_chars=800)
    assert len(chunks) > 1
    assert all(c["heading_path"] == "Big" for c in chunks)


def test_chunk_markdown_strips_date_metadata_line():
    # prepare_foundry_dataset "(Doküman tarihi: ...)" satırı ekliyor — bu tek başına
    # işe yaramaz chunk üretmemeli (gerçek "create model" hatasının kaynağıydı).
    md = (
        "# Compile Hugging Face models\n"
        "(Doküman tarihi: 05/29/2026)\n\n"
        "# Compile Hugging Face models\n\n"
        "Use Olive to convert a Hugging Face model into ONNX."
    )
    chunks = chunk_markdown(md, max_chars=800)
    # tarih-only chunk oluşmamalı
    assert not any(c["content"].strip().startswith("(Doküman tarihi:") for c in chunks)
    # gerçek içerik chunk'ı olmalı
    assert any("Use Olive" in c["content"] for c in chunks)
