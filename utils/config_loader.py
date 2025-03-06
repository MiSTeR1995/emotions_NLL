import os
import toml

class ConfigLoader:
    """Класс для загрузки и обработки конфигурации из `config.toml`."""

    def __init__(self, config_path="config.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"⚠️ Ошибка: Файл конфигурации `{config_path}` не найден!")

        self.config = toml.load(config_path)

        # 🔹 Основные пути
        self.split = self.config.get("split", "dev")  # "train", "dev", "test"
        self.base_dir = self.config.get("base_dir", "E:/MELD")  # Корневой каталог данных
        self.csv_path = self.config.get("csv_path", "{base_dir}/MELD.Raw/meld_{split}_labels.csv").format(base_dir=self.base_dir, split=self.split)
        self.wav_dir = self.config.get("wav_dir", "{base_dir}/wavs/{split}").format(base_dir=self.base_dir, split=self.split)
        self.video_dir = self.config.get("video_dir", "{base_dir}/MELD.Raw/{split}_splits").format(base_dir=self.base_dir, split=self.split)

        # 🔹 Датасет
        self.modalities = self.config.get("modalities", ["audio"])  # Какие модальности использовать
        self.emotion_columns = self.config.get("emotion_columns", ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"])

        # 🔹 DataLoader
        self.batch_size = self.config.get("dataloader", {}).get("batch_size", 1)
        self.num_workers = self.config.get("dataloader", {}).get("num_workers", 1)
        self.shuffle = self.config.get("dataloader", {}).get("shuffle", True)

        # 🔹 Аудио параметры
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 16000)
        self.wav_length = self.config.get("audio", {}).get("wav_length", 2)

        # 🔹 Параметры сохранения аудио
        self.save_processed_audio = self.config.get("audio_saving", {}).get("save_processed_audio", False)
        self.audio_output_dir = self.config.get("audio_saving", {}).get("audio_output_dir", "{base_dir}/output_wavs").format(base_dir=self.base_dir)

        # 🔹 Текстовые параметры
        self.text_source = self.config.get("text", {}).get("source", "whisper")  # "csv" или "whisper"
        self.text_column = self.config.get("text", {}).get("text_column", "text")  # Название колонки с текстом
        self.whisper_model = self.config.get("text", {}).get("whisper_model", "small")  # Whisper model
        self.max_text_tokens = self.config.get("text", {}).get("max_tokens", 15)  # Ограничение длины текста (по словам)

        # 🔹 Логируем конфиг только в главном процессе
        if __name__ == "__main__":
            self.log_config()

    def log_config(self):
        """Выводит текущую конфигурацию в консоль (только в главном процессе)."""
        print("\n🔹 Конфигурация загружена:")
        print(f"   Split: {self.split}")
        print(f"   CSV Path: {self.csv_path}")
        print(f"   WAV Dir: {self.wav_dir}")
        print(f"   Video Dir: {self.video_dir}")
        print(f"   Modalities: {self.modalities}")
        print(f"   Batch Size: {self.batch_size}, Num Workers: {self.num_workers}, Shuffle: {self.shuffle}")
        print(f"   Audio: Sample Rate = {self.sample_rate}, Length = {self.wav_length}s")

        print(f"   Text Source: {self.text_source}, Text Column: {self.text_column}")
        print(f"   Whisper Model: {self.whisper_model}, Max Tokens: {self.max_text_tokens}")

        # 🔹 Добавляем инфу про сохранение аудио
        if self.save_processed_audio:
            print(f"   Audio Saving: ✅ Включено, файлы сохраняются в `{self.audio_output_dir}`")
        else:
            print(f"   Audio Saving: ❌ Выключено")

    def update_config(self, **kwargs):
        """Позволяет изменять параметры на лету."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"✅ Параметр `{key}` обновлён: {value}")
            else:
                print(f"⚠️ Ошибка: Параметр `{key}` не найден в конфиге!")

    def show_config(self):
        """Выводит текущую конфигурацию (можно вызывать вручную)."""
        self.log_config()

# 🔹 Создаём глобальный объект конфигурации
config = ConfigLoader()

# Если запускаем файл напрямую, выводим конфиг
if __name__ == "__main__":
    config.show_config()
