from foundry_local_sdk import Configuration, FoundryLocalManager

CHAT_MODEL_ALIAS = "phi-3.5-mini"

# Modelin eğitim verisinde kesinlikle olmayan, uydurma bir "bilgi"
CONTEXT = "Efe'nin favori rengi mordur. Efe aynı zamanda haftada üç kez yüzmeye gider."
QUESTION = "Efe'nin favori rengi nedir?"

SYSTEM_PROMPT_GROUNDED = (
    "Sen bir yardımcı asistansın. Sadece aşağıda verilen BAĞLAM'daki bilgiyi kullanarak cevap ver. "
    "Eğer cevap bağlamda yoksa 'Bu bilgiye sahip değilim' de. Bağlam dışına çıkma, tahmin yürütme."
)


def main():
    config = Configuration(app_name="LocalRagAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(CHAT_MODEL_ALIAS)
    model.load()
    chat_client = model.get_chat_client()

    print("=" * 60)
    print("TEST 1: BAĞLAMSIZ (modele sadece soru soruluyor)")
    print("=" * 60)
    print(f"Soru: {QUESTION}\n")
    response_no_context = chat_client.complete_chat([
        {"role": "user", "content": QUESTION}
    ])
    print("Model yanıtı:")
    print(response_no_context.choices[0].message.content)

    print("\n" + "=" * 60)
    print("TEST 2: BAĞLAMLI (RAG'in yaptığı gibi — context enjekte edildi)")
    print("=" * 60)
    print(f"Bağlam: {CONTEXT}")
    print(f"Soru: {QUESTION}\n")
    response_with_context = chat_client.complete_chat([
        {"role": "system", "content": SYSTEM_PROMPT_GROUNDED},
        {"role": "user", "content": f"BAĞLAM: {CONTEXT}\n\nSORU: {QUESTION}"},
    ])
    print("Model yanıtı:")
    print(response_with_context.choices[0].message.content)


if __name__ == "__main__":
    main()
