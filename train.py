import torch
from torch.utils.data import DataLoader
from data_loading.dataset_multimodal import DatasetMultiModal
from processing.audio_processing import AudioProcessor
from utils.config_loader import config

def custom_collate_fn(batch):
    """Удаляет `None` из батча и пропускает пустые батчи."""
    batch = [b for b in batch if b is not None]

    if batch:
        # Теперь b['audio'] будет тензором, а не кортежем
        audio_batch = torch.stack([b['audio'] for b in batch])
        batch_dict = {
            "audio": audio_batch,
            "text": [b['text'] for b in batch],
            "label": torch.stack([b['label'] for b in batch])
        }
        return batch_dict

    return None



if __name__ == "__main__":
    print("🚀 Запуск тренировки с конфигурацией:")
    config.show_config()

    # 🔹 Создаём датасет (без препроцессора, чтобы сначала создать `audio_class_map`, только если split == "train")
    dataset = DatasetMultiModal(
        csv_path=config.csv_path,
        wav_dir=config.wav_dir,
        emotion_columns=config.emotion_columns,
        split=config.split,
        modalities=config.modalities,
        text_source=config.text_source,
        text_column=config.text_column
    )

    # 🔹 Если `train`, создаём аудио-препроцессор с `audio_class_map`
    audio_processor = AudioProcessor(
        sample_rate=config.sample_rate,
        wav_length=config.wav_length,
        save_processed_audio=config.save_processed_audio,
        output_dir=config.audio_output_dir,
        split=config.split,
        audio_class_map=dataset.audio_class_map,
        whisper_model=config.whisper_model,
        max_text_tokens=config.max_text_tokens  # <-- Новое ограничение для текста
    )

    # 🔹 Передаём процессор в датасет
    dataset.audio_processor = audio_processor

    # 🔹 Создаём DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    print("\n🚀 Начинаем загрузку данных из DataLoader...")

    for batch in dataloader:
        if batch is None:
            print("⚠️ Пропущен пустой батч")
            continue

        print(f"\n🔹 Batch загружен:")
        print(f"   - Batch Audio shape: {batch['audio'].shape if batch['audio'] is not None else 'None'}")
        print(f"   - Batch Emotion vector shape: {batch['label'].shape}")

        # 🔹 Выводим one-hot эмоции
        for i, emotions in enumerate(batch['label']):
            print(f"🎭 Эмоция для примера {i}: {emotions.tolist()}")

        # 🔹 Выводим текстовые транскрипции
        for i, text in enumerate(batch['text']):
            text_display = text if text.strip() else "⚠️ [Пустая транскрипция]"
            print(f"📝 Текст для примера {i}: {text_display}")

        break  # Только один батч для теста
