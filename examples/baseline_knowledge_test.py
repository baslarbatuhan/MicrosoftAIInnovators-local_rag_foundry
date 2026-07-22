"""
Taban bilgi testi: phi-3.5-mini'ye Foundry Local sorularını BAĞLAMSIZ sorar.

Amaç: modelin eğitim kesiminden (≈Ekim 2023) sonra çıkan bir ürün hakkında
tek başına ne bildiğini (bilmediğini/uydurduğunu) belgelemek. Bu çıktı,
"neden RAG?" sorusunun canlı kanıtı — Demo Day'de RAG'li cevaplarla
yan yana gösterilecek.

Çalıştırma (proje kökünden): python examples/baseline_knowledge_test.py
Çıktı ayrıca knowledge_bases/foundry/eval/baseline_no_rag.txt dosyasına kaydedilir.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.core import load_chat_client  # noqa: E402

QUESTIONS = [
    "What is Microsoft Foundry Local?",
    "How do I install Foundry Local on Windows?",
    "Which CLI command lists the models in the Foundry Local catalog?",
    "How do I generate embeddings with the Foundry Local Python SDK?",
    "What is the difference between the current and the legacy Foundry Local SDK?",
]

# Kasıtlı olarak nötr system prompt — grounding/reddetme talimatı YOK.
# Modelin kendi bilgisiyle baş başa kaldığında ne yaptığını görmek istiyoruz.
NEUTRAL_PROMPT = "You are a helpful assistant. Answer the user's question."


def main():
    print("Model yükleniyor...")
    chat_client = load_chat_client()

    lines = ["TABAN BİLGİ TESTİ — phi-3.5-mini, BAĞLAMSIZ (RAG yok)",
             "Model kesimi ~Ekim 2023; Foundry Local dokümanları 2025-2026 tarihli.", ""]
    for question in QUESTIONS:
        response = chat_client.complete_chat([
            {"role": "system", "content": NEUTRAL_PROMPT},
            {"role": "user", "content": question},
        ])
        answer = response.choices[0].message.content.strip()
        block = f"SORU: {question}\nCEVAP: {answer}\n" + "-" * 70
        print("\n" + block)
        lines.append(block)

    out_dir = os.path.join("knowledge_bases", "foundry", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_no_rag.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nKayıt: {out_path}")


if __name__ == "__main__":
    main()
