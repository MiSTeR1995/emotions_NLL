# data_loading/dataset_multimodal.py

import os
import random
import logging
import torch
import torchaudio
import whisper
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DatasetMultiModal(Dataset):
    """
    Мультимодальный датасет для аудио, текста и эмоций (онлайновая версия).

    При каждом вызове __getitem__:
      - Загружает WAV (по video_name в CSV).
      - (train) Если короткое — склеивает со случайным другим WAV той же эмоции.
      - Падит нулями, если всё ещё короче, чтобы достичь длины wav_length.
      - Если нужно — вызывает Whisper (примерно в следующих случаях):
        1) Если действительно была склейка (новый сэмпл) => получаем текст заново.
        2) Если текст в CSV отсутствует.
        3) (Опционально) если dev/test и нет текста, можно тоже вызвать Whisper.
      - Возвращает словарь { "audio": ..., "text": ..., "label": ... }.
    """

    def __init__(
        self,
        csv_path,
        wav_dir,
        emotion_columns,
        split="train",
        sample_rate=16000,
        wav_length=2,
        whisper_model="tiny",
        max_text_tokens=15,
        text_column="text",
        use_whisper_for_nontrain_if_no_text=True,
        whisper_device="cuda",
        subset_size=0
    ):
        """
        :param csv_path: Путь к CSV-файлу (video_name, emotion_columns, text?).
        :param wav_dir: Папка с аудиофайлами (video_name.wav).
        :param emotion_columns: Список колонок эмоций, напр. ["neutral","happy",...].
        :param split: "train", "dev" или "test".
        :param sample_rate: Целевая частота дискретизации (по умолчанию 16k).
        :param wav_length: Целевая длина аудио в секундах (напр. 2).
        :param whisper_model: Название модели (tiny, base, small, ...).
        :param max_text_tokens: Максимальное число слов (tokens), оставляемых в тексте.
        :param text_column: Название колонки с текстом в CSV.
        :param use_whisper_for_nontrain_if_no_text: Нужно ли Whisper для dev/test, если нет текста.
        :param whisper_device: "cuda" или "cpu" — куда загружать модель Whisper.
        """
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.target_samples = int(wav_length * sample_rate)
        self.emotion_columns = emotion_columns
        self.whisper_model_name = whisper_model
        self.max_text_tokens = max_text_tokens
        self.text_column = text_column
        self.use_whisper_for_nontrain_if_no_text = use_whisper_for_nontrain_if_no_text
        self.whisper_device = whisper_device

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл CSV не найден: {csv_path}")

        df = pd.read_csv(csv_path)

        # Если subset_size > 0, ограничиваемся первыми N записями
        if subset_size > 0:
            df = df.head(subset_size)
            logging.info(f"[DatasetMultiModal] Используем только первые {len(df)} записей (subset_size={subset_size}).")

        # Проверяем, что все указанные колонки эмоций есть
        missing = [c for c in emotion_columns if c not in df.columns]
        if missing:
            raise ValueError(f"В CSV отсутствуют необходимые колонки эмоций: {missing}")

        # Проверяем, что папка с WAV существует
        if not os.path.exists(wav_dir):
            raise ValueError(f"Ошибка: директория с аудио {wav_dir} не существует!")

        # Собираем список строк (video_name, эмоции, text)
        self.rows = []
        for i, rowi in df.iterrows():
            audio_path = os.path.join(wav_dir, f"{rowi['video_name']}.wav")
            if not os.path.exists(audio_path):
                continue

            # Находим доминирующую эмоцию (макс. значение)
            emotion_values = rowi[self.emotion_columns].values.astype(float)
            max_idx = np.argmax(emotion_values)
            emotion_label = self.emotion_columns[max_idx]

            # Текст из CSV (если есть)
            csv_text = ""
            if self.text_column in rowi and isinstance(rowi[self.text_column], str):
                csv_text = rowi[self.text_column]

            self.rows.append({
                "audio_path": audio_path,
                "label": emotion_label,
                "csv_text": csv_text
            })

        # Карта {audio_path: emotion_label} для поиска файлов склейки
        self.audio_class_map = {}
        for entry in self.rows:
            self.audio_class_map[entry["audio_path"]] = entry["label"]

        logging.info(f"[DatasetMultiModal] Сплит={split}, всего строк: {len(self.rows)}")

        # Загружаем Whisper-модель один раз
        logging.info(f"Инициализация Whisper: модель={whisper_model}, устройство={whisper_device}")
        self.whisper_model = whisper.load_model(whisper_model, device=whisper_device)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        """
        Загружает и обрабатывает один элемент датасета (онлайново).
        """
        row = self.rows[index]
        audio_path = row["audio_path"]
        label_name = row["label"]
        csv_text = row["csv_text"]

        # Формируем one-hot вектор эмоций
        label_vec = self.emotion_to_vector(label_name)

        # Шаг 1. Загружаем аудио
        waveform, sr = self.load_audio(audio_path)
        if waveform is None:
            return None

        original_length = waveform.shape[1]
        logging.debug(f"Исходная длина {os.path.basename(audio_path)}: "
                      f"{original_length / sr:.2f} сек")

        # Шаг 2. Склейка (train), если короткое
        was_merged = False
        if self.split == "train" and original_length < self.target_samples:
            needed = self.target_samples - original_length
            add_path = self.get_suitable_audio(label_name, audio_path, needed)
            if add_path:
                add_wf, add_sr = self.load_audio(add_path)
                if add_wf is not None:
                    logging.debug(f"Склеиваем {os.path.basename(audio_path)} "
                                  f"с {os.path.basename(add_path)}")
                    waveform = torch.cat((waveform, add_wf), dim=1)
                    was_merged = True

        # Шаг 3. Если всё ещё короткое => паддинг
        curr_len = waveform.shape[1]
        if curr_len < self.target_samples:
            pad_size = self.target_samples - curr_len
            logging.debug(f"Паддинг {os.path.basename(audio_path)}: +{pad_size} сэмплов")
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # Обрезаем, если вдруг длиннее
        waveform = waveform[:, :self.target_samples]

        final_len = waveform.shape[1]
        logging.debug(f"Финальная длина {os.path.basename(audio_path)}: "
                      f"{final_len / sr:.2f} сек; was_merged={was_merged}")

        # Шаг 4. Получаем текст (Whisper или из CSV)
        text_final = self.get_text_logic(csv_text, was_merged, waveform)

        # Возвращаем итог
        return {
            "audio": waveform,
            "label": label_vec,
            "text": text_final
        }

    def get_text_logic(self, csv_text, was_merged, waveform):
        """
        Определяет, откуда брать текст:
          - Если was_merged=True => Whisper
          - Иначе, если csv_text не пуст => берём csv_text
          - Иначе, если split=train => Whisper
          - Иначе, если dev/test => Whisper только если use_whisper_for_nontrain_if_no_text=True
        """
        if was_merged:
            logging.debug("Текст: вызываем Whisper (так как была склейка).")
            return self.run_whisper(waveform)

        # Если не было склейки
        if csv_text.strip():
            logging.debug("Текст: используем csv_text (не пуст).")
            return csv_text
        else:
            # csv_text пуст
            if self.split == "train":
                logging.debug("Текст: пустое CSV, split=train => Whisper.")
                return self.run_whisper(waveform)
            else:
                # dev/test
                if self.use_whisper_for_nontrain_if_no_text:
                    logging.debug("Текст: пустое CSV, dev/test => Whisper.")
                    return self.run_whisper(waveform)
                else:
                    logging.debug("Текст: пустое CSV, dev/test => без Whisper (пустая строка).")
                    return ""

    def load_audio(self, path):
        """Загружает аудио, ресэмплит при необходимости."""
        if not os.path.exists(path):
            logging.warning(f"Файл отсутствует: {path}")
            return None, None
        try:
            wf, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                transform = torchaudio.transforms.Resample(sr, self.sample_rate)
                wf = transform(wf)
                sr = self.sample_rate
            return wf, sr
        except Exception as e:
            logging.error(f"Ошибка загрузки {path}: {e}")
            return None, None

    def get_suitable_audio(self, label_name, exclude_path, min_needed):
        """
        Ищет другой файл с той же эмоцией, у которого длина >= min_needed (в сэмплах).
        Возвращает путь, либо None, если не найдено.
        """
        candidates = [
            p for p, lbl in self.audio_class_map.items()
            if lbl == label_name and p != exclude_path
        ]
        valid = []
        for path in candidates:
            try:
                info = torchaudio.info(path)
                length = info.num_frames
                sr_ = info.sample_rate
                if sr_ != self.sample_rate:
                    ratio = sr_ / self.sample_rate
                    eq_len = int(length / ratio)
                    if eq_len >= min_needed:
                        valid.append(path)
                else:
                    if length >= min_needed:
                        valid.append(path)
            except:
                pass

        if not valid:
            return None
        return random.choice(valid)

    def run_whisper(self, waveform):
        """
        Вызывает Whisper на загруженной модели,
        обрезает результат до max_text_tokens слов.
        """
        arr = waveform.squeeze().cpu().numpy()
        try:
            result = self.whisper_model.transcribe(arr, fp16=False)
            text = result["text"].strip()
            tokens = text.split()
            truncated = " ".join(tokens[:self.max_text_tokens])
            return truncated
        except Exception as e:
            logging.error(f"Whisper ошибка: {e}")
            return ""

    def emotion_to_vector(self, label_name):
        """
        Преобразует название эмоции в one-hot вектор
        (по списку self.emotion_columns).
        """
        v = np.zeros(len(self.emotion_columns), dtype=np.float32)
        if label_name in self.emotion_columns:
            idx = self.emotion_columns.index(label_name)
            v[idx] = 1.0
        return torch.tensor(v, dtype=torch.float32)
