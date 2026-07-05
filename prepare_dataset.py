"""
Tek seferlik hazırlık scripti: Kaggle'daki "single-topic-rag-evaluation-dataset"
içinden 5 dokümanı seçip documents/ klasörüne .txt olarak yazar, ilgili
soru setlerini (cevaplanabilir/cevaplanamaz) eval/ klasörüne filtreler.

Kaynak: kaggle.com/datasets/samuelmatsuoharris/single-topic-rag-evaluation-dataset
Yeniden çalıştırmak güvenlidir (var olan dosyaların üzerine yazar).
"""
import csv
import os
import sys

import kagglehub

csv.field_size_limit(sys.maxsize)

# İşleyeceğimiz dokümanlar: (document_index, dosya adı)
SELECTED_DOCS = {
    0: "bullet_kin.txt",
    1: "dnd_underground_session.txt",
    2: "stici_note_rag_blog.txt",
    3: "llmware_readme.txt",
    4: "marimo_recipes.txt",
}

DOCUMENTS_DIR = "documents"
EVAL_DIR = "eval"


def main():
    dataset_path = kagglehub.dataset_download("samuelmatsuoharris/single-topic-rag-evaluation-dataset")
    print(f"Dataset yolu: {dataset_path}")

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # 1. Dokümanları .txt olarak yaz (ilk satır: kaynak URL, sonra boş satır, sonra metin)
    with open(os.path.join(dataset_path, "documents.csv"), encoding="utf-8") as f:
        docs = list(csv.DictReader(f))

    for idx, filename in SELECTED_DOCS.items():
        doc = docs[idx]
        out_path = os.path.join(DOCUMENTS_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(f"Kaynak: {doc['source_url']}\n\n")
            out.write(doc["text"])
        print(f"  yazildi: {out_path} ({len(doc['text'])} karakter)")

    # 2. Soru setlerini sadece seçtiğimiz dokümanlara göre filtrele
    for csv_name in ["single_passage_answer_questions.csv", "multi_passage_answer_questions.csv", "no_answer_questions.csv"]:
        with open(os.path.join(dataset_path, csv_name), encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = [r for r in reader if int(r["document_index"]) in SELECTED_DOCS]

        for row in rows:
            row["source_file"] = SELECTED_DOCS[int(row["document_index"])]

        out_path = os.path.join(EVAL_DIR, csv_name)
        with open(out_path, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames + ["source_file"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  yazildi: {out_path} ({len(rows)} soru)")


if __name__ == "__main__":
    main()
