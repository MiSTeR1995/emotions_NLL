import torchaudio
import torch
import os
import whisper
import unicodedata
import random
import numpy as np

class TextProcessor:
    """Класс для обработки текста (извлечение, обрезка по токенам)."""

    def __init__(self, max_tokens=15, whisper_model="tiny"):
        """
        :param max_tokens: Максимальное количество токенов (слов) в тексте.
        :param whisper_model: Whisper-модель для извлечения текста.
        """
        self.max_tokens = max_tokens
        self.whisper_model = whisper.load_model(whisper_model)

    def extract_text(self, audio_path):
        """Извлекает текст из аудиофайла с Whisper и обрезает по токенам."""
        if not os.path.exists(audio_path):
            print(f"⚠️ Файл для распознавания не найден: {audio_path}")
            return ""

        try:
            result = self.whisper_model.transcribe(audio_path)
            text = self.clean_text(result["text"])
            return self.trim_text(text)

        except Exception as e:
            print(f"⚠️ Ошибка распознавания текста `{audio_path}`: {e}")
            return ""

    def extract_text_from_waveform(self, waveform):
        """
        Извлекает текст из переданного аудиосигнала (waveform).
        Если пришёл тензор PyTorch, конвертируем его в NumPy (и удаляем размерность канала, если нужно).
        """
        # Новая проверка: если waveform — это PyTorch-тензор, переводим в NumPy
        if isinstance(waveform, torch.Tensor):
            # Сожмём размерность канала (B=1) в (samples,) и переведём в NumPy
            waveform = waveform.squeeze(0).cpu().numpy()

        if not isinstance(waveform, np.ndarray):
            print(f"⚠️ Ошибка: ожидался np.ndarray или torch.Tensor, получено {type(waveform)}")
            return ""

        try:
            result = self.whisper_model.transcribe(waveform)
            text = self.clean_text(result["text"])
            return self.trim_text(text)

        except Exception as e:
            print(f"⚠️ Ошибка распознавания текста из аудиосигнала: {e}")
            return ""

    def trim_text(self, text):
        """Обрезает текст по количеству токенов (слов)."""
        tokens = text.split()
        return " ".join(tokens[:self.max_tokens])

    @staticmethod
    def clean_text(text):
        """Очищает текст от непечатаемых символов и нормализует юникод."""
        text = unicodedata.normalize("NFKC", text)
        text = text.encode("ascii", "ignore").decode("utf-8")  # Убираем не-ASCII символы
        return text.strip()


class AudioProcessor:
    """Класс для загрузки, обработки аудио и извлечения текста."""

    def __init__(self, sample_rate=16000, wav_length=2, save_processed_audio=False,
                 output_dir="output_wavs", split="train", audio_class_map=None,
                 whisper_model="tiny", max_text_tokens=15):
        """
        :param sample_rate: Частота дискретизации (Гц).
        :param wav_length: Длина аудио (в секундах).
        :param save_processed_audio: Сохранять ли обработанные аудио.
        :param output_dir: Папка для сохранения обработанных файлов.
        :param split: Тип выборки ("train", "dev", "test").
        :param audio_class_map: Словарь {аудио_файл: класс_эмоции} для склейки.
        :param whisper_model: Whisper-модель для обработки текста.
        :param max_text_tokens: Максимальное количество слов в транскрипции.
        """
        self.sample_rate = sample_rate
        self.wav_length = wav_length * sample_rate
        self.save_processed_audio = save_processed_audio
        self.output_dir = output_dir
        self.split = split
        self.audio_class_map = audio_class_map if audio_class_map else {}

        # Текстовый процессор (Whisper)
        self.text_processor = TextProcessor(
            max_tokens=max_text_tokens,
            whisper_model=whisper_model
        )

        # Проверяем доступные аудиобэкенды
        available_backends = torchaudio.list_audio_backends()
        self.audio_backend = "sox_io" if "sox_io" in available_backends else \
                             "soundfile" if "soundfile" in available_backends else None

        if not self.audio_backend:
            print("⚠️ Внимание: Нет доступных аудиобэкендов. torchaudio.load() может не работать.")

    def load_audio(self, path):
        """
        Загружает аудиофайл, анализирует его длину, при необходимости склеивает (train)
        или падит нулями (dev/test). Возвращает только waveform (тензор).
        """
        if not os.path.exists(path):
            print(f"⚠️ Файл отсутствует: {path}")
            return None

        try:
            waveform, sample_rate = torchaudio.load(path, backend=self.audio_backend)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {path}: {e}")
            return None

        original_length = waveform.shape[1]
        print(f"🔹 Исходная длина аудио `{os.path.basename(path)}`: {original_length / sample_rate:.2f} сек")

        if self.split == "train":
            # 🔄 **Только train: Склейка коротких аудиофайлов**
            if original_length < self.wav_length:
                print(
                    f"🔄 Аудио `{os.path.basename(path)}` короче "
                    f"{self.wav_length / self.sample_rate:.2f} сек, ищем файл для склейки..."
                )
                add_file = self.get_suitable_audio(path, self.wav_length - original_length)
                if add_file:
                    try:
                        add_waveform, _ = torchaudio.load(add_file, backend=self.audio_backend)
                        if add_waveform is not None:
                            waveform = torch.cat((waveform, add_waveform), dim=1)
                            print(f"🔗 Склеено с файлом `{os.path.basename(add_file)}`")
                    except Exception as e:
                        print(f"⚠️ Ошибка загрузки файла для склейки `{add_file}`: {e}")

        else:
            # 🔹 **Для `dev` и `test`: Паддинг нулями**
            if original_length < self.wav_length:
                pad_size = self.wav_length - original_length
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
                print(f"🔹 Аудио `{os.path.basename(path)}`: добавлено {pad_size / sample_rate:.2f} сек нулей.")

        # Обрезаем до нужной длины
        waveform = waveform[:, :self.wav_length]

        # ❗️ Возвращаем только тензор waveform (без sample_rate)
        return waveform

    def get_suitable_audio(self, original_path, min_needed_length):
        """Ищет файл, который при склейке с оригиналом даст длину ≥ `wav_length`."""
        emotion_label = self.audio_class_map.get(original_path)
        if not emotion_label:
            return None

        candidates = [
            path for path, label in self.audio_class_map.items()
            if label == emotion_label and path != original_path
        ]

        valid_files = []
        for path in candidates:
            if os.path.exists(path):
                waveform, _ = torchaudio.load(path)
                if waveform.shape[1] >= min_needed_length:
                    valid_files.append(path)

        return random.choice(valid_files) if valid_files else None
