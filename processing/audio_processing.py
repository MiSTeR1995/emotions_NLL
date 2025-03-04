import torchaudio
import torch
import os
import whisper
import unicodedata

class AudioProcessor:
    """Класс для загрузки, обработки аудио и извлечения текста."""

    def __init__(self, sample_rate=16000, wav_length=2, text_source="whisper", text_column="text",
                 model="base", save_processed_audio=False, output_dir="output_wavs"):
        """
        :param sample_rate: Частота дискретизации (Гц).
        :param wav_length: Длина аудио (в секундах).
        :param text_source: Способ получения текста ("whisper" или "csv").
        :param text_column: Название колонки с текстом (если используется "csv").
        :param model: Название модели Whisper ("tiny", "base", "small", "medium", "large").
        :param save_processed_audio: Сохранять ли обработанные аудио (из config).
        :param output_dir: Папка для сохранения аудио.
        """
        self.sample_rate = sample_rate
        self.wav_length = wav_length * sample_rate  # Переводим секунды в сэмплы
        self.text_source = text_source
        self.text_column = text_column
        self.model_name = model  # Сохраняем название модели

        self.save_processed_audio = save_processed_audio
        self.output_dir = output_dir

        # Загружаем Whisper, если выбрано "whisper"
        self.whisper_model = whisper.load_model(self.model_name) if text_source == "whisper" else None

        # Проверяем доступные аудиобэкенды
        available_backends = torchaudio.list_audio_backends()
        self.audio_backend = "sox_io" if "sox_io" in available_backends else "soundfile" if "soundfile" in available_backends else None

        if not self.audio_backend:
            print("⚠️ Внимание: Нет доступных аудиобэкендов. torchaudio.load() может не работать.")

    def load_audio(self, path):
        """Загружает аудиофайл, обрезает его и сохраняет в output_dir (если включено)."""
        if not os.path.exists(path):
            print(f"⚠️ Файл отсутствует: {path}")
            return None

        if not path.endswith(".wav"):
            print(f"⚠️ Неподдерживаемый формат: {path}")
            return None

        if not self.audio_backend:
            print(f"⚠️ Ошибка: Нет доступного аудиобэкенда для загрузки {path}")
            return None

        try:
            waveform, sample_rate = torchaudio.load(path, backend=self.audio_backend)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки {path}: {e}")
            return None

        # Выводим исходную длину аудио
        original_length = waveform.shape[1]
        print(f"🔹 Исходная длина аудио: {original_length / sample_rate:.2f} сек ({original_length} сэмплов)")

        # Обрезаем или паддим аудио до нужной длины
        if waveform.shape[1] < self.wav_length:
            pad_size = self.wav_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:, :self.wav_length]

        # Выводим новую длину аудио
        processed_length = waveform.shape[1]
        print(f"✅ Обрезанное аудио: {processed_length / sample_rate:.2f} сек ({processed_length} сэмплов)")

        # 🔹 Сохраняем обработанное аудио, если включено в конфиге
        if self.save_processed_audio:
            os.makedirs(self.output_dir, exist_ok=True)  # Создаём папку, если её нет
            output_path = os.path.join(self.output_dir, os.path.basename(path))
            torchaudio.save(output_path, waveform, sample_rate)
            print(f"📁 Аудио сохранено: {output_path}")

        return waveform

    def get_text(self, path, row):
        """Возвращает текст из CSV или извлекает с Whisper."""
        if self.text_source == "csv":
            return row[self.text_column] if self.text_column in row else ""
        elif self.text_source == "whisper":
            return self.transcribe_audio(path)
        return ""

    def transcribe_audio(self, path):
        """Распознаёт текст из аудиофайла с Whisper и очищает его."""
        if not self.whisper_model:
            return ""

        if not os.path.exists(path):
            print(f"⚠️ Файл не найден для распознавания: {path}")
            return ""

        try:
            result = self.whisper_model.transcribe(path)
            text = result["text"]
            return self.clean_text(text)  # Очищаем текст перед возвратом

        except Exception as e:
            print(f"⚠️ Ошибка распознавания текста `{path}`: {e}")
            return ""

    @staticmethod
    def clean_text(text):
        """Очищает текст от непечатаемых символов и нормализует юникод."""
        text = unicodedata.normalize("NFKC", text)  # Приводим к стандартному юникоду
        text = text.encode("ascii", "ignore").decode("utf-8")  # Убираем все не-ASCII символы
        return text.strip()
