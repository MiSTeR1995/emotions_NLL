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
    """Собирает список образцов в единый батч, отбрасывая None."""
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

def make_dataset_and_loader(config, split: str):
    """
    Функция, которая создаёт Dataset и DataLoader для указанного сплита.
    Читает config.csv_path / config.wav_dir как шаблоны, подставляет split.
    Возвращает (dataset, dataloader).
    """
    print(split)
    # Формируем пути, подставляя {split} и {base_dir}
    csv_path = config.csv_path.format(base_dir=config.base_dir, split=split)
    wav_dir  = config.wav_dir.format(base_dir=config.base_dir,  split=split)

    print(csv_path)
    # Создаём датасет
    dataset = DatasetMultiModal(
        csv_path           = csv_path,
        wav_dir            = wav_dir,
        emotion_columns    = config.emotion_columns,
        split              = split,
        sample_rate        = config.sample_rate,
        wav_length         = config.wav_length,
        whisper_model      = config.whisper_model,
        text_column        = config.text_column,
        use_whisper_for_nontrain_if_no_text = config.use_whisper_for_nontrain_if_no_text,
        whisper_device     = config.whisper_device,
        subset_size        = config.subset_size,
        merge_probability  = config.merge_probability
    )

    # Создаём DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size = config.batch_size,
        shuffle    = config.shuffle,
        num_workers= config.num_workers,
        collate_fn = custom_collate_fn
    )

    return dataset, dataloader

def main():
    os.makedirs("logs", exist_ok=True)
    datestr  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join("logs", f"train_log_{datestr}.txt")
    setup_logger(logging.DEBUG, log_file=log_file)

    logging.info("🚀 === Запуск тренировки со всеми сплитами ===")

    # Загружаем конфиг (без поля split)
    config = ConfigLoader("config.toml")
    config.show_config()   # Просто напечатает остальные поля

    # Фиксируем seed, если указан
    if config.random_seed > 0:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        logging.info(f"🔒 Фиксируем random seed: {config.random_seed}")
    else:
        logging.info("🔓 Random seed не фиксирован (0).")

    # Создаём экстракторы эмбеддингов ОДИН раз — можно переиспользовать
    audio_extractor = AudioEmbeddingExtractor(config)
    text_extractor  = TextEmbeddingExtractor(config)

    # Пройдёмся по списку сплитов
    for split_name in ["train", "dev", "test"]:
        logging.info(f"\n=== Обрабатываем split='{split_name}' ===")

        # Создаём датасет и DataLoader для этого сплита
        dataset, dataloader = make_dataset_and_loader(config, split_name)

        # Допустим, хотим просто 1 эпоху для примера (или 2)
        for epoch in range(1):
            logging.info(f"--- Эпоха {epoch}, split={split_name} ---")

            for i, batch in enumerate(dataloader):
                if batch is None:
                    continue

                audio = batch["audio"]   # (B, 1, samples)
                labels= batch["label"]   # (B, num_emotions)
                texts = batch["text"]    # список строк

                logging.info(f"[Epoch={epoch}, split={split_name}, Batch={i}] audio={audio.shape}, labels={labels.shape}")

                # Извлекаем эмбеддинги
                audio_emb = audio_extractor.extract(audio, sample_rate=config.sample_rate)
                text_emb  = text_extractor.extract(texts)

                logging.info(f"Audio emb shape: {audio_emb.shape}")
                logging.info(f"Text  emb shape: {text_emb.shape}")

                # Здесь ваша логика обучения / валидации / теста
                # if split_name=='train':
                #     optimizer.zero_grad()
                #     loss=...
                #     loss.backward()
                #     optimizer.step()
                # elif split_name=='dev':
                #     ...
                # elif split_name=='test':
                #     ...

                if i==0:
                    logging.info(f"   Пример текста[0]: {texts[0]}")
                    logging.info(f"   Пример Audio Emb[0]: {audio_emb[0][:5]}")
                    logging.info(f"   Пример Text  Emb[0]:  {text_emb[0][:5]}")

    logging.info("✅ Готово! Обработка train/dev/test завершена.")

if __name__ == "__main__":
    main()
