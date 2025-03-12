# train.py

import logging
import os
import datetime
import torch
import random
from torch.utils.data import DataLoader

from utils.logger_setup import setup_logger
from utils.config_loader import ConfigLoader
from data_loading.dataset_multimodal import DatasetMultiModal

def main():
    # Создаём папку для логов (если не существует)
    os.makedirs("logs", exist_ok=True)

    # Генерируем имя файла с датой/временем
    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join("logs", f"train_log_{datestr}.txt")

    # Инициализируем логгер: цветные логи в консоль + запись в файл
    setup_logger(logging.DEBUG, log_file=log_file)
    logging.info("🚀 === Запуск on-the-fly тренировки ===")

    # Загружаем конфиг
    config = ConfigLoader("config.toml")
    config.show_config()

    # Фиксируем сид, если указан в конфиге
    if config.random_seed is not None:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        logging.info(f"🔒 Фиксируем random seed: {config.random_seed}")
    else:
        logging.info("🔓 random seed НЕ фиксирован (None).")

    # Создаём датасет
    dataset = DatasetMultiModal(
        csv_path=config.csv_path,
        wav_dir=config.wav_dir,
        emotion_columns=config.emotion_columns,
        split=config.split,
        sample_rate=config.sample_rate,
        wav_length=config.wav_length,
        whisper_model=config.whisper_model,
        max_text_tokens=config.max_text_tokens,
        text_column=config.text_column,
        use_whisper_for_nontrain_if_no_text=config.use_whisper_for_nontrain_if_no_text,
        whisper_device=config.whisper_device,
        subset_size=config.subset_size
    )

    # Создаём DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn
    )

    # Пример цикла обучения
    for epoch in range(2):
        logging.info(f"=== Эпоха {epoch} ===")
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue

            audio = batch["audio"]
            labels = batch["label"]
            texts = batch["text"]

            logging.info(f"[Epoch={epoch} Batch={i}] audio_shape={audio.shape}, label_shape={labels.shape}")
            if texts:
                logging.info(f"Пример текста[0]: {texts[0]}")

    logging.info("✅ Тренировка завершена.")

def custom_collate_fn(batch):
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

if __name__ == "__main__":
    main()
