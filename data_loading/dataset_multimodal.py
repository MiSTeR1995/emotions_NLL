import os
from data_loading.dataset_base import BaseDataset
import torch
import numpy as np

class DatasetMultiModal(BaseDataset):
    """Мультимодальный датасет для аудио, текста и эмоций."""

    def __init__(self, csv_path, wav_dir, emotion_columns, split="train", modalities=None,
                 audio_processor=None, text_source="whisper", text_column="text"):
        super().__init__(csv_path, emotion_columns, text_column)

        if not os.path.exists(wav_dir):
            raise ValueError(f"Ошибка: директория с аудио {wav_dir} не существует!")

        self.wav_dir = wav_dir
        self.split = split
        self.modalities = modalities if modalities else ["audio"]
        self.audio_processor = audio_processor
        self.text_source = text_source
        self.text_column = text_column

        # 🔹 Создаём соответствие {путь к аудио: класс эмоции} ТОЛЬКО ДЛЯ train
        self.audio_class_map = {}
        if self.split == "train":
            print("🔄 Создаём `audio_class_map` для train...")
            self.audio_class_map = {
                os.path.join(self.wav_dir, f"{row['video_name']}.wav"): self.get_emotion_label(row)
                for _, row in self.df.iterrows()
            }

    def __getitem__(self, idx):
        """Загружает один элемент (аудио, текст, эмоции)."""
        row = self.df.iloc[idx]

        # 🔹 Формируем путь к аудиофайлу
        audio_path = os.path.join(self.wav_dir, f"{row['video_name']}.wav")

        if not os.path.exists(audio_path):
            print(f"⚠️ Файл не найден: {audio_path}")
            return None

        # 🔹 Загружаем аудио
        if self.audio_processor:
            waveform = self.audio_processor.load_audio(audio_path)
        else:
            waveform = None

        if waveform is None:
            print(f"⚠️ Ошибка загрузки аудио `{audio_path}`")
            return None

        # 🔹 Получаем текст
        text = self.get_text(row, waveform)

        # 🔹 Загружаем эмоции
        emotion_vector = self.get_emotion_vector(row)

        return {
            "audio": waveform,   # Возвращаем *тензор* (waveform)
            "text": text,
            "label": emotion_vector
        }

    def get_text(self, row, waveform):
        """
        Возвращает текст:
        - Для `train` - всегда Whisper.
        - Для `dev` и `test` - либо Whisper, либо из CSV, в зависимости от text_source.
        """
        if self.split == "train" or self.text_source == "whisper":
            return self.audio_processor.text_processor.extract_text_from_waveform(waveform) if self.audio_processor else ""
        elif self.text_source == "csv":
            text = row[self.text_column] if self.text_column in row else ""
            return self.audio_processor.text_processor.trim_text(text)
        return ""

    def get_emotion_label(self, row):
        """Возвращает название эмоции (например, 'happy', 'sad')."""
        emotion_values = row[self.emotion_columns].values.astype(float)
        max_index = np.argmax(emotion_values)  # Индекс самой сильной эмоции
        return self.emotion_columns[max_index]
