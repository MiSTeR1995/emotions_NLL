import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """Базовый класс для работы с мультимодальными датасетами (аудио, видео, текст)."""

    def __init__(self, csv_path, emotion_columns, text_column=None):
        """
        :param csv_path: Путь к CSV-файлу с метками.
        :param emotion_columns: Список колонок с эмоциями (обязательный).
        :param text_column: Название колонки с текстом (если есть в CSV).
        """
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл {csv_path} не найден!")

        if not emotion_columns:
            raise ValueError("Список колонок эмоций (emotion_columns) обязателен!")

        self.csv_path = csv_path
        self.emotion_columns = emotion_columns
        self.text_column = text_column  # Добавляем поддержку текстовой колонки

        # 🔹 Загружаем CSV
        self.df = pd.read_csv(csv_path)

        # 🔹 Проверяем, что все указанные колонки эмоций есть в CSV
        missing_columns = [col for col in self.emotion_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"В CSV отсутствуют следующие колонки эмоций: {missing_columns}")

        # 🔹 Оставляем только `video_name`, эмоции и текстовую колонку (если она есть)
        keep_columns = ["video_name"] + self.emotion_columns

        if self.text_column:
            if self.text_column in self.df.columns:
                keep_columns.append(self.text_column)
            else:
                print(f"⚠️ Предупреждение: Колонка `{self.text_column}` отсутствует в CSV!")

        self.df = self.df[keep_columns]  # Оставляем только нужные колонки

        print(f"✅ Загружено {len(self.df)} записей из `{csv_path}`")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Определяет интерфейс (будет реализован в подклассе)."""
        raise NotImplementedError("Метод __getitem__() должен быть реализован в подклассе")

    def get_emotion_vector(self, row):
        """Преобразует метки эмоций в one-hot вектор (float32)."""
        emotion_values = row[self.emotion_columns].values.astype(float)  # Приводим к float
        return torch.tensor(emotion_values, dtype=torch.float32)
