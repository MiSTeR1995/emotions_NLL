import os
from data_loading.dataset_base import BaseDataset
from processing.audio_processing import AudioProcessor
import torch

class DatasetMultiModal(BaseDataset):
    """Мультимодальный датасет для аудио, текста и эмоций."""

    def __init__(self, csv_path, wav_dir, emotion_columns, modalities=None, audio_processor=None, text_source="whisper", text_column="text"):
        """
        :param csv_path: Путь к CSV с метками.
        :param wav_dir: Папка с аудиофайлами.
        :param emotion_columns: Список колонок с эмоциями (обязательный).
        :param modalities: Какие модальности использовать (список ['audio']).
        :param audio_processor: Препроцессор для аудио и текста.
        :param text_source: Источник текста ("whisper" или "csv").
        :param text_column: Колонка с текстом в CSV (если source = "csv").
        """
        super().__init__(csv_path, emotion_columns)

        if not os.path.exists(wav_dir):
            raise ValueError(f"Ошибка: директория с аудио {wav_dir} не существует!")

        self.wav_dir = wav_dir
        self.modalities = modalities if modalities else ["audio"]
        self.audio_processor = audio_processor
        self.text_source = text_source
        self.text_column = text_column

    def __getitem__(self, idx):
        """Загружает один элемент (аудио, текст, эмоции)."""
        row = self.df.iloc[idx]

        # 🔹 Формируем путь к аудиофайлу на основе `video_name`
        audio_path = os.path.join(self.wav_dir, f"{row['video_name']}.wav")

        if not os.path.exists(audio_path):
            print(f"⚠️ Файл не найден: {audio_path}")
            return None

        # 🔹 Загружаем аудио
        audio = self.audio_processor.load_audio(audio_path) if self.audio_processor else None
        if audio is None:
            print(f"⚠️ Ошибка загрузки аудио `{audio_path}`")
            return None

        # 🔹 Получаем текст
        text = self.get_text(audio_path, row)

        # 🔹 Загружаем эмоции
        emotion_vector = self.get_emotion_vector(row)

        return {
            "audio": audio,
            "text": text,
            "label": emotion_vector
        }

    def get_text(self, audio_path, row):
        """Возвращает текст из CSV или извлекает с Whisper."""
        if self.text_source == "csv":
            return row[self.text_column] if self.text_column in self.df.columns else ""
        elif self.text_source == "whisper":
            return self.audio_processor.transcribe_audio(audio_path) if self.audio_processor else ""
        return ""
