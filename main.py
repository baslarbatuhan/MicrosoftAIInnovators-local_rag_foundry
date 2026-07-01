from foundry_local_sdk import Configuration, FoundryLocalManager

MODEL_ALIAS = "phi-3.5-mini"


def main():
    # 1. SDK'yı başlat (uygulama adı zorunlu, dosya/klasör adı olarak kullanılıyor)
    config = Configuration(app_name="LocalRagAssistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 2. Kataloğdan modeli bul
    model = manager.catalog.get_model(MODEL_ALIAS)
    if model is None:
        raise RuntimeError(f"Model bulunamadı: {MODEL_ALIAS}")

    # 3. Model cihazda yoksa indir (ilk çalıştırmada birkaç dakika sürebilir)
    if not model.is_cached:
        print(f"'{MODEL_ALIAS}' indiriliyor...")
        model.download(progress_callback=lambda pct: print(f"  %{pct:.1f}", end="\r"))
        print()

    # 4. Modeli belleğe yükle
    print("Model yükleniyor...")
    model.load()

    # 5. Chat client al ve basit bir soru sor
    chat_client = model.get_chat_client()
    response = chat_client.complete_chat([
        {"role": "user", "content": "Hello, world"}
    ])

    print("\nModel yanıtı:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
