import torch
from torch.utils.data import DataLoader
from data_loading.dataset_multimodal import DatasetMultiModal
from processing.audio_processing import AudioProcessor
from utils.config_loader import config

def custom_collate_fn(batch):
    """Удаляет `None` из батча и пропускает пустые батчи."""
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch) if batch else None  # Если батч пустой, возвращаем `None`

if __name__ == "__main__":
    print("🚀 Запуск тренировки с конфигурацией:")
    config.show_config()

    # 🔹 Создаём аудио-препроцессор с выбором модели Whisper
    audio_processor = AudioProcessor(
        sample_rate=config.sample_rate,
        wav_length=config.wav_length,
        text_source=config.text_source,
        text_column=config.text_column,
        model=config.whisper_model,
        save_processed_audio=config.save_processed_audio,
        output_dir=config.audio_output_dir
    )

    # 🔹 Создаём датасет
    dataset = DatasetMultiModal(
        csv_path=config.csv_path,
        wav_dir=config.wav_dir,
        emotion_columns=config.emotion_columns,
        modalities=config.modalities,
        audio_processor=audio_processor,
        text_source=config.text_source,
        text_column=config.text_column
    )

    # 🔹 Создаём DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    for batch in dataloader:
        if batch is None:
            print("⚠️ Пропущен пустой батч")
            continue

        print(f"Batch Audio shape: {batch['audio'].shape if batch['audio'] is not None else 'None'}")
        print(f"Batch Emotion vector shape: {batch['label'].shape}")

        # 🔹 Выводим one-hot эмоции
        for i, emotions in enumerate(batch['label']):
            print(f"🎭 Эмоции для примера {i}: {emotions.tolist()}")

        # 🔹 Выводим текстовые транскрипции
        for i, text in enumerate(batch['text']):
            print(f"📝 Текст для примера {i}: {text}")

        break  # Только один батч для теста
