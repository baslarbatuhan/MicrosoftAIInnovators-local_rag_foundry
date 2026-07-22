"""cosine_similarity() ve read_document() birim testleri."""
import math

from rag.ingest import read_document
from rag.retrieval import cosine_similarity


# --- cosine_similarity ---

def test_identical_vectors_score_one():
    v = [0.5, -1.0, 2.0]
    assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-9)


def test_orthogonal_vectors_score_zero():
    assert math.isclose(cosine_similarity([1, 0], [0, 1]), 0.0, abs_tol=1e-9)


def test_opposite_vectors_score_minus_one():
    assert math.isclose(cosine_similarity([1, 2], [-1, -2]), -1.0, rel_tol=1e-9)


def test_scale_invariance():
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    scaled = [x * 10 for x in b]
    assert math.isclose(cosine_similarity(a, b), cosine_similarity(a, scaled), rel_tol=1e-9)


# --- read_document ---

def test_read_document_parses_kaynak_header(tmp_path):
    doc = tmp_path / "sample.txt"
    doc.write_text("Kaynak: https://example.com/page\n\nBody first line.\nSecond line.",
                   encoding="utf-8")
    source_url, body = read_document(str(doc))
    assert source_url == "https://example.com/page"
    assert body.startswith("Body first line.")
    assert "Kaynak:" not in body


def test_read_document_without_header_uses_filename(tmp_path):
    doc = tmp_path / "plain.txt"
    doc.write_text("Just plain content without a header.", encoding="utf-8")
    source_url, body = read_document(str(doc))
    assert source_url == "plain.txt"
    assert body == "Just plain content without a header."
