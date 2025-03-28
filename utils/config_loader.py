# utils/config_loader.py

import os
import toml
import logging

class ConfigLoader:
    """
    Класс для загрузки и обработки конфигурации из `config.toml`.
    """

    def __init__(self, config_path="config.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации `{config_path}` не найден!")

        self.config = toml.load(config_path)

        # ---------------------------
        # Общие параметры
        # ---------------------------
        self.split = self.config.get("split")
        self.base_dir = self.config.get("base_dir", "E:/MELD")

        # ---------------------------
        # Пути к данным
        # ---------------------------
        # self.csv_path = self.config.get("csv_path", "{base_dir}/MELD.Raw/meld_{split}_labels.csv") \
        #     .format(base_dir=self.base_dir, split=self.split)
        # self.wav_dir = self.config.get("wav_dir", "{base_dir}/wavs/{split}") \
        #     .format(base_dir=self.base_dir, split=self.split)
        # self.video_dir = self.config.get("video_dir", "{base_dir}/MELD.Raw/{split}_splits") \
        #     .format(base_dir=self.base_dir, split=self.split)

        self.csv_path = self.config.get("csv_path", "{base_dir}/MELD.Raw/meld_{split}_labels.csv")
        self.wav_dir = self.config.get("wav_dir", "{base_dir}/wavs/{split}")
        self.video_dir = self.config.get("video_dir", "{base_dir}/MELD.Raw/{split}_splits")

        # ---------------------------
        # Эмоции, модальности
        # ---------------------------
        self.modalities = self.config.get("modalities", ["audio"])
        self.emotion_columns = self.config.get("emotion_columns",
                                               ["neutral","happy","sad","anger","surprise","disgust","fear"])

        # ---------------------------
        # DataLoader
        # ---------------------------
        dataloader_cfg = self.config.get("dataloader", {})
        self.batch_size = dataloader_cfg.get("batch_size", 1)
        self.num_workers = dataloader_cfg.get("num_workers", 0)
        self.shuffle = dataloader_cfg.get("shuffle", True)

        # ---------------------------
        # Аудио
        # ---------------------------
        audio_cfg = self.config.get("audio", {})
        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.wav_length = audio_cfg.get("wav_length", 2)

        # ---------------------------
        # Whisper / Текст
        # ---------------------------
        text_cfg = self.config.get("text", {})
        self.text_source = text_cfg.get("source", "csv")
        self.text_column = text_cfg.get("text_column", "text")
        self.whisper_model = text_cfg.get("whisper_model", "tiny")
        self.max_text_tokens = text_cfg.get("max_tokens", 15)
        self.whisper_device = text_cfg.get("whisper_device", "cuda")
        self.use_whisper_for_nontrain_if_no_text = text_cfg.get("use_whisper_for_nontrain_if_no_text", True)

        # ---------------------------
        # Параметры для тренировки
        # ---------------------------
        train_cfg = self.config.get("train", {})
        self.random_seed = train_cfg.get("random_seed", None)
        self.subset_size = train_cfg.get("subset_size", 0)
        self.merge_probability = train_cfg.get("merge_probability", 0.5)

        # ---------------------------
        # Embeddings
        # ---------------------------
        emb_cfg = self.config.get("embeddings", {})
        self.audio_model_name = emb_cfg.get("audio_model", "amiriparian/ExHuBERT")
        self.text_model_name  = emb_cfg.get("text_model",  "jinaai/jina-embeddings-v3")

        self.audio_embedding_dim = emb_cfg.get("audio_embedding_dim", 1024)
        self.text_embedding_dim  = emb_cfg.get("text_embedding_dim",  1024)

        self.audio_pooling = emb_cfg.get("audio_pooling", None)
        self.text_pooling  = emb_cfg.get("text_pooling",  None)

        self.max_tokens = emb_cfg.get("max_tokens", 256)
        self.max_audio_frames = emb_cfg.get("max_audio_frames", 16000)

        self.emb_device = emb_cfg.get("device", "cuda")
        self.emb_normalize = emb_cfg.get("normalize_output", True)

        if __name__ == "__main__":
            self.log_config()

    def log_config(self):
        logging.info("=== CONFIGURATION ===")
        logging.info(f"Split: {self.split}")
        logging.info(f"Base Dir: {self.base_dir}")
        logging.info(f"CSV Path: {self.csv_path}")
        logging.info(f"WAV Dir: {self.wav_dir}")
        logging.info(f"Emotion columns: {self.emotion_columns}")
        logging.info(f"Sample Rate={self.sample_rate}, Wav Length={self.wav_length}s")
        logging.info(f"Whisper Model={self.whisper_model}, Device={self.whisper_device}, MaxTokens={self.max_text_tokens}")
        logging.info(f"use_whisper_for_nontrain_if_no_text={self.use_whisper_for_nontrain_if_no_text}")
        logging.info(f"DataLoader: batch_size={self.batch_size}, num_workers={self.num_workers}, shuffle={self.shuffle}")
        logging.info(f"Random Seed={self.random_seed}")
        logging.info(f"Merge Probability={self.merge_probability}")

        # Логируем embeddings
        logging.info("--- Embeddings Config ---")
        logging.info(f"Audio Model: {self.audio_model_name}, Text Model: {self.text_model_name}")
        logging.info(f"Audio dim={self.audio_embedding_dim}, Text dim={self.text_embedding_dim}")
        logging.info(f"Audio pooling={self.audio_pooling}, Text pooling={self.text_pooling}")
        logging.info(f"Max tokens={self.max_tokens}, Max audio frames={self.max_audio_frames}")
        logging.info(f"Emb device={self.emb_device}, Normalize={self.emb_normalize}")

    def show_config(self):
        self.log_config()
