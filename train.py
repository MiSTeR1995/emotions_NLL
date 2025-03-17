# train.py

import logging
import torch
import random
import os
import datetime
from torch.utils.data import DataLoader

from utils.logger_setup import setup_logger
from utils.config_loader import ConfigLoader
from data_loading.dataset_multimodal import DatasetMultiModal
from data_loading.feature_extractor import AudioEmbeddingExtractor, TextEmbeddingExtractor

def custom_collate_fn(batch):
    """
    Собирает список образцов в единый батч, отбрасывая None.
    """
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    audios = [b["audio"] for b in batch]
    audio_tensor = torch.stack(audios)

    labels = [b["label"] for b in batch]
    label_tensor = torch.stack(labels)

    texts = [b["text"] for b in batch]

    return {
        "audio": audio_tensor,
        "label": label_tensor,
        "text": texts
    }

def main():
    # Создаем папку для логов и уникальное имя файла с датой/временем
    os.makedirs("logs", exist_ok=True)
    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join("logs", f"train_log_{datestr}.txt")
    setup_logger(logging.DEBUG, log_file=log_file)

    logging.info("🚀 === Запуск on-the-fly тренировки ===")

    # Загружаем конфиг
    config = ConfigLoader("config.toml")
    config.show_config()

    # Фиксируем seed, если указан (seed > 0)
    if config.random_seed > 0:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        logging.info(f"🔒 Фиксируем random seed: {config.random_seed}")
    else:
        logging.info("🔓 Random seed не фиксирован (0).")

    # Создаем датасет
    dataset = DatasetMultiModal(
        csv_path=config.csv_path,
        wav_dir=config.wav_dir,
        emotion_columns=config.emotion_columns,
        split=config.split,
        sample_rate=config.sample_rate,
        wav_length=config.wav_length,
        whisper_model=config.whisper_model,
        text_column=config.text_column,
        use_whisper_for_nontrain_if_no_text=config.use_whisper_for_nontrain_if_no_text,
        whisper_device=config.whisper_device,
        subset_size=config.subset_size,
        merge_probability=config.merge_probability
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    # === Создаём экстракторы эмбеддингов ===
    audio_extractor = AudioEmbeddingExtractor(config)
    text_extractor  = TextEmbeddingExtractor(config)

# Упрощённый цикл "обучения"
    for epoch in range(2):
        logging.info(f"=== Эпоха {epoch} ===")

        for i, batch in enumerate(dataloader):
            if batch is None:
                continue

            audio = batch["audio"]   # shape: (B, 1, samples)
            labels = batch["label"]  # shape: (B, num_emotions)
            texts = batch["text"]    # список строк длины B

            logging.info(f"[Epoch={epoch} Batch={i}] audio_shape={audio.shape}, label_shape={labels.shape}")

            # Извлечём аудио-эмбеддинги
            audio_emb = audio_extractor.extract(audio, sample_rate=config.sample_rate)
            logging.info(f"Audio emb shape: {audio_emb.shape}")

            # Извлечём текст-эмбеддинги
            text_emb = text_extractor.extract(texts)
            logging.info(f"Text emb shape: {text_emb.shape}")    # (B, text_embedding_dim)

            # Для демонстрации выведем лишь один пример:
            if i == 0:
                logging.info(f"Пример текста[0]: {texts[0]}")
                logging.info(f"Пример Audio Emb[0]: {audio_emb[0][:5]}")  # первые 5 чисел
                logging.info(f"Пример Text  Emb[0]: {text_emb[0][:5]}")

        # Конец эпохи
    logging.info("✅ Тренировка завершена.")

if __name__ == "__main__":
    main()
