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
    Мультимодальный датасет для аудио, текста и эмоций (он‑the‑fly версия).

    При каждом вызове __getitem__:
      - Загружает WAV по video_name из CSV.
      - Для обучающей выборки (split="train"):
            Если аудио короче target_samples, проверяем, выбрали ли мы этот файл для склейки
            (по merge_probability). Если да – выполняется "chain merge":
            выбирается один или несколько дополнительных файлов того же класса, даже если один кандидат длиннее,
            и итоговое аудио затем обрезается до точной длины.
      - Если итоговое аудио всё ещё меньше target_samples, выполняется паддинг нулями.
      - Текст выбирается так:
            • Если аудио было merged (склеено) – вызывается Whisper для получения нового текста.
            • Если merge не происходило и CSV-текст не пуст – используется CSV-текст.
            • Если CSV-текст пустой – для train (или, при условии, для dev/test) вызывается Whisper.
      - Возвращает словарь { "audio": waveform, "label": label_vector, "text": text_final }.
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
        text_column="text",
        use_whisper_for_nontrain_if_no_text=True,
        whisper_device="cuda",
        subset_size=0,
        merge_probability=1.0  # <-- Новый параметр: доля от ОБЩЕГО числа файлов
    ):
        """
        :param csv_path: Путь к CSV-файлу (с колонками video_name, emotion_columns, возможно text).
        :param wav_dir: Папка с аудиофайлами (имя файла: video_name.wav).
        :param emotion_columns: Список колонок эмоций, например ["neutral", "happy", "sad", ...].
        :param split: "train", "dev" или "test".
        :param sample_rate: Целевая частота дискретизации (например, 16000).
        :param wav_length: Целевая длина аудио в секундах.
        :param whisper_model: Название модели Whisper ("tiny", "base", "small", ...).
        :param max_text_tokens: (Не используется) – ограничение на число токенов.
        :param text_column: Название колонки с текстом в CSV.
        :param use_whisper_for_nontrain_if_no_text: Если True, для dev/test при отсутствии CSV-текста вызывается Whisper.
        :param whisper_device: "cuda" или "cpu" – устройство для модели Whisper.
        :param subset_size: Если > 0, используется только первые N записей из CSV (для отладки).
        :param merge_probability: Процент (0..1) от всего числа файлов, которые будут склеиваться, если они короче.
        """
        super().__init__()
        self.split = split
        self.sample_rate = sample_rate
        self.target_samples = int(wav_length * sample_rate)
        self.emotion_columns = emotion_columns
        self.whisper_model_name = whisper_model
        self.text_column = text_column
        self.use_whisper_for_nontrain_if_no_text = use_whisper_for_nontrain_if_no_text
        self.whisper_device = whisper_device
        self.merge_probability = merge_probability

        # Загружаем CSV
        if not os.path.exists(csv_path):
            raise ValueError(f"Ошибка: файл CSV не найден: {csv_path}")
        df = pd.read_csv(csv_path)
        if subset_size > 0:
            df = df.head(subset_size)
            logging.info(f"[DatasetMultiModal] Используем только первые {len(df)} записей (subset_size={subset_size}).")

        # Проверяем наличие всех колонок эмоций
        missing = [c for c in emotion_columns if c not in df.columns]
        if missing:
            raise ValueError(f"В CSV отсутствуют необходимые колонки эмоций: {missing}")

        # Проверяем существование папки с аудио
        if not os.path.exists(wav_dir):
            raise ValueError(f"Ошибка: директория с аудио {wav_dir} не существует!")
        self.wav_dir = wav_dir

        # Собираем список строк: для каждой записи получаем путь к аудио, label и CSV-текст (если есть)
        self.rows = []
        for i, rowi in df.iterrows():
            audio_path = os.path.join(wav_dir, f"{rowi['video_name']}.wav")
            if not os.path.exists(audio_path):
                continue
            # Определяем доминирующую эмоцию (максимальное значение)
            emotion_values = rowi[self.emotion_columns].values.astype(float)
            max_idx = np.argmax(emotion_values)
            emotion_label = self.emotion_columns[max_idx]

            # Извлекаем текст из CSV (если есть)
            csv_text = ""
            if self.text_column in rowi and isinstance(rowi[self.text_column], str):
                csv_text = rowi[self.text_column]

            self.rows.append({
                "audio_path": audio_path,
                "label": emotion_label,
                "csv_text": csv_text
            })

        # Создаем карту для поиска файлов по эмоции
        self.audio_class_map = {entry["audio_path"]: entry["label"] for entry in self.rows}

        logging.info("📊 Анализ распределения файлов по эмоциям:")
        emotion_counts = {emotion: 0 for emotion in set(self.audio_class_map.values())}
        for path, emotion in self.audio_class_map.items():
            emotion_counts[emotion] += 1
        for emotion, count in emotion_counts.items():
            logging.info(f"🎭 Эмоция '{emotion}': {count} файлов.")

        logging.info(f"[DatasetMultiModal] Сплит={split}, всего строк: {len(self.rows)}")

        # === Процентное семплирование ===
        total_files = len(self.rows)
        num_to_merge = int(total_files * self.merge_probability)

        # Определим, какие файлы "короткие" (могут нуждаться в склейке)
        self.mergable_files = []
        self.mergable_files = [
            row["audio_path"]  # вместо целого dict берём строку
            for row in self.rows
            if self._is_too_short(row["audio_path"])
        ]
        short_count = len(self.mergable_files)

        # Если коротких файлов больше нужного числа, выберем случайные. Иначе все короткие.
        if short_count > num_to_merge:
            # Выбираем случайно нужное количество
            self.files_to_merge = set(random.sample(self.mergable_files, num_to_merge))
        else:
            # Склеим все короткие
            self.files_to_merge = set(self.mergable_files)

        logging.info(f"🔗 Всего файлов: {total_files}, нужно склеить: {num_to_merge} ({self.merge_probability*100:.0f}%)")
        logging.info(f"🔗 Коротких файлов: {short_count}, выбрано для склейки: {len(self.files_to_merge)}")

        # Инициализируем Whisper-модель один раз
        logging.info(f"Инициализация Whisper: модель={whisper_model}, устройство={whisper_device}")
        self.whisper_model = whisper.load_model(whisper_model, device=whisper_device)

    def _is_too_short(self, audio_path):
        """
        Проверяем, является ли файл короче target_samples.
        """
        try:
            info = torchaudio.info(audio_path)
            length = info.num_frames
            sr_ = info.sample_rate
            # переводим длину в "эквивалент self.sample_rate"
            if sr_ != self.sample_rate:
                ratio = sr_ / self.sample_rate
                eq_len = int(length / ratio)
            else:
                eq_len = length
            return eq_len < self.target_samples
        except Exception as e:
            logging.warning(f"Ошибка _is_too_short({audio_path}): {e}")
            return False

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        """
        Загружает и обрабатывает один элемент датасета (он‑the‑fly).
        """
        row = self.rows[index]
        audio_path = row["audio_path"]
        label_name = row["label"]
        csv_text = row["csv_text"]

        # Преобразуем label в one-hot вектор
        label_vec = self.emotion_to_vector(label_name)

        # Шаг 1. Загружаем аудио
        waveform, sr = self.load_audio(audio_path)
        if waveform is None:
            return None

        orig_len = waveform.shape[1]
        logging.debug(f"Исходная длина {os.path.basename(audio_path)}: {orig_len/sr:.2f} сек")

        was_merged = False
        merged_texts = [csv_text]  # Тексты исходного файла + добавленных

        # Шаг 2. Для train, если аудио короче target_samples, проверяем:
        #        попал ли данный row в files_to_merge?
        if self.split == "train" and row["audio_path"] in self.files_to_merge:
            # chain merge
            current_length = orig_len
            used_candidates = set()

            while current_length < self.target_samples:
                needed = self.target_samples - current_length
                candidate = self.get_suitable_audio(label_name, exclude_path=audio_path, min_needed=needed)
                if candidate is None or candidate in used_candidates:
                    break
                used_candidates.add(candidate)
                add_wf, add_sr = self.load_audio(candidate)
                if add_wf is None:
                    break
                logging.debug(f"Склейка: добавляем {os.path.basename(candidate)} (необходимых сэмплов: {needed})")
                waveform = torch.cat((waveform, add_wf), dim=1)
                current_length = waveform.shape[1]
                was_merged = True

                # Получаем текст второго файла (если есть в CSV)
                add_csv_text = next((r["csv_text"] for r in self.rows if r["audio_path"] == candidate), "")
                merged_texts.append(add_csv_text)

                logging.debug(f"📜 Текст первого файла: {csv_text}")
                logging.debug(f"📜 Текст добавленного файла: {add_csv_text}")
        else:
            # Если файл не в списке "должны склеить" или сплит не train, пропускаем chain-merge
            logging.debug("Файл не выбран для склейки (или не train), пропускаем chain merge.")

        # Шаг 3. Если итоговая длина меньше target_samples, паддинг нулями
        curr_len = waveform.shape[1]
        if curr_len < self.target_samples:
            pad_size = self.target_samples - curr_len
            logging.debug(f"Паддинг {os.path.basename(audio_path)}: +{pad_size} сэмплов")
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # Шаг 4. Обрезаем аудио до target_samples (если вышло больше)
        waveform = waveform[:, :self.target_samples]
        logging.debug(f"Финальная длина {os.path.basename(audio_path)}: {waveform.shape[1]/sr:.2f} сек; was_merged={was_merged}")

        # Шаг 5. Получаем текст
        if was_merged:
            logging.debug("📝 Текст: аудио было merged – вызываем Whisper.")
            text_final = self.run_whisper(waveform)
            logging.debug(f"🆕 Whisper предсказал: {text_final}")
        else:
            if csv_text.strip():
                logging.debug("Текст: используем CSV-текст (не пуст).")
                text_final = csv_text
            else:
                if self.split == "train" or self.use_whisper_for_nontrain_if_no_text:
                    logging.debug("Текст: CSV пустой – вызываем Whisper.")
                    text_final = self.run_whisper(waveform)
                else:
                    logging.debug("Текст: CSV пустой и не вызываем Whisper для dev/test.")
                    text_final = ""

        return {
            "audio": waveform,
            "label": label_vec,
            "text": text_final
        }

    def load_audio(self, path):
        """
        Загружает аудио по указанному пути и ресэмплирует его до self.sample_rate, если необходимо.
        """
        if not os.path.exists(path):
            logging.warning(f"Файл отсутствует: {path}")
            return None, None
        try:
            wf, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                wf = resampler(wf)
                sr = self.sample_rate
            return wf, sr
        except Exception as e:
            logging.error(f"Ошибка загрузки {path}: {e}")
            return None, None

    def get_suitable_audio(self, label_name, exclude_path, min_needed):
        """
        Ищет аудиофайл с той же эмоцией, длина которого >= min_needed (в сэмплах).
        """
        candidates = [p for p, lbl in self.audio_class_map.items() if lbl == label_name and p != exclude_path]
        logging.debug(f"🔍 Найдено {len(candidates)} кандидатов для класса '{label_name}'")

        valid = []
        for path in candidates:
            try:
                info = torchaudio.info(path)
                length = info.num_frames
                sr_ = info.sample_rate
                eq_len = int(length * (self.sample_rate / sr_)) if sr_ != self.sample_rate else length
                if eq_len >= min_needed:
                    valid.append((eq_len, path))
            except Exception as e:
                logging.warning(f"⚠ Ошибка чтения {path}: {e}")

        logging.debug(f"✅ Подходящих файлов: {len(valid)} (из {len(candidates)})")

        if not valid:
            return None  # Нет подходящих файлов

        random.shuffle(valid)
        return random.choice(valid)[1]

    def run_whisper(self, waveform):
        """
        Вызывает Whisper на аудиосигнале и возвращает полный текст (без ограничения по количеству слов).
        """
        arr = waveform.squeeze().cpu().numpy()
        try:
            result = self.whisper_model.transcribe(arr, fp16=False)
            text = result["text"].strip()
            return text
        except Exception as e:
            logging.error(f"Whisper ошибка: {e}")
            return ""

    def emotion_to_vector(self, label_name):
        """
        Преобразует название эмоции в one-hot вектор (torch.tensor).
        """
        v = np.zeros(len(self.emotion_columns), dtype=np.float32)
        if label_name in self.emotion_columns:
            idx = self.emotion_columns.index(label_name)
            v[idx] = 1.0
        return torch.tensor(v, dtype=torch.float32)
