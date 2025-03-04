import torch
import torchvision.io as io
import os

try:
    import av
except ImportError:
    av = None

class VideoProcessor:
    """Класс для загрузки и обработки видео."""

    def __init__(self, frame_size=(224, 224), num_frames=16):
        self.frame_size = frame_size
        self.num_frames = num_frames

        # Проверяем, установлен ли PyAV
        if av is None:
            print("⚠️ Внимание: PyAV не установлен! Видео не будет загружаться.")

    def load_video(self, path):
        """Загружает и обрабатывает видеофайл."""
        if av is None:
            print("⚠️ Ошибка: PyAV не установлен, невозможно загрузить видео.")
            return None

        if not os.path.exists(path):
            print(f"⚠️ Файл отсутствует: {path}")
            return None

        try:
            video, audio, info = io.read_video(path, pts_unit="sec")  # 📌 Используем pts_unit="sec"
        except Exception as e:
            print(f"⚠️ Ошибка загрузки видео {path}: {e}")
            return None

        # Берём первые num_frames кадров (если не хватает, паддим)
        if video.shape[0] < self.num_frames:
            pad_frames = self.num_frames - video.shape[0]
            pad = torch.zeros((pad_frames, *video.shape[1:]))
            video = torch.cat([video, pad], dim=0)
        else:
            video = video[:self.num_frames]

        return video
