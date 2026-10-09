import os
import time
import av
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Any

from tqdm import tqdm
from faster_whisper import WhisperModel
from logger import Logger


SUPPORTED_EXTENSIONS = {'.mp3', '.mp4', '.wav', '.m4a', '.flac', '.ogg', '.wma'}


def _format_time(seconds: float) -> str:
    """
    Конвертирует секунды в формат ЧЧ:ММ:СС или ММ:СС.

    :param seconds: Время в секундах.
    :returns: Строка с форматированным временем.

    Example:
        >>> _format_time(3665)
        '01:01:05'
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def _format_duration(seconds: float) -> str:
    """
    Форматирует длительность в читаемый вид.

    :param seconds: Длительность в секундах.
    :returns: Строка вида '1ч 23мин 45с'.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f"{h}ч")
    if m > 0:
        parts.append(f"{m}мин")
    parts.append(f"{s}с")
    return " ".join(parts)


def _format_file_size(size_bytes: int) -> str:
    """
    Форматирует размер файла в читаемый вид.

    :param size_bytes: Размер в байтах.
    :returns: Строка с размером и единицей измерения.
    """
    for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} ТБ"


def _get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Собирает метаданные о файле.

    :param file_path: Путь к файлу.
    :returns: Словарь с метаданными (имя, размер, расширение).
    """
    return {
        "file_name": os.path.basename(file_path),
        "file_size": os.path.getsize(file_path),
        "file_ext": os.path.splitext(file_path)[1].lower(),
    }


def _load_and_resample_audio(file_path: str) -> Tuple[np.ndarray, float]:
    """
    Читает аудио и конвертирует в 16kHz mono float32 для Whisper.

    :param file_path: Путь к аудиофайлу.
    :returns: Кортеж (waveform, duration_in_seconds).
    """
    Logger.debug(f"Чтение и ресемплинг аудио: {os.path.basename(file_path)}", name="transcriber")
    container = av.open(file_path)
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=16000)

    audio_data = []
    for frame in container.decode(stream):
        frame.pts = None
        for resampled_frame in resampler.resample(frame):
            audio_data.append(resampled_frame.to_ndarray().flatten())

    waveform = np.concatenate(audio_data)
    duration = len(waveform) / 16000.0
    return waveform, duration


def _transcribe_waveform(waveform: np.ndarray, model: WhisperModel, duration: float, language: str = "ru") -> List[Dict[str, Any]]:
    """
    Выполняет транскрибацию аудио и возвращает список сегментов с отображением прогресса.

    :param waveform: Массив аудиоданных (16kHz, mono, float32).
    :param model: Экземпляр загруженной модели WhisperModel.
    :param duration: Общая длительность аудио в секундах (для прогресс-бара).
    :param language: Код языка для распознавания.
    :returns: Список словарей вида {'start': float, 'end': float, 'text': str}.
    """
    segments, _ = model.transcribe(
        waveform,
        language=language,
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200)
    )
    
    result = []
    last_end = 0.0
    with tqdm(total=duration, unit="сек", desc="Транскрибация") as pbar:
        for segment in segments:
            result.append({
                "start": segment.start, 
                "end": segment.end, 
                "text": segment.text.strip()
            })
            # Обновляем прогресс на основе конца текущего сегмента
            update_step = max(0.0, segment.end - last_end)
            pbar.update(update_step)
            last_end = segment.end
            
    return result


def _group_segments(segments: List[Dict[str, Any]], interval_seconds: int) -> List[Dict[str, Any]]:
    """
    Группирует сегменты транскрибации по заданным временным интервалам.

    :param segments: Список сегментов с ключами start, end, text.
    :param interval_seconds: Длительность интервала группировки в секундах.
    :returns: Список сгруппированных блоков.
    """
    if not segments:
        return []

    buckets = defaultdict(lambda: {"start": None, "end": None, "text_parts": []})

    for segment in segments:
        interval_idx = int(segment["start"] // interval_seconds)
        bucket = buckets[interval_idx]

        if bucket["start"] is None:
            bucket["start"] = segment["start"]
        bucket["end"] = segment["end"]
        bucket["text_parts"].append(segment["text"])

    grouped = []
    for idx in sorted(buckets.keys()):
        bucket = buckets[idx]
        text = " ".join(bucket["text_parts"])
        text = " ".join(text.split())  # очистка от множественных пробелов
        grouped.append({
            "interval_start": bucket["start"],
            "interval_end": bucket["end"],
            "text": text
        })
    return grouped


def _find_audio_files(directory: str) -> List[str]:
    """
    Находит все поддерживаемые аудиофайлы в указанной директории.

    :param directory: Путь к директории.
    :returns: Отсортированный список полных путей к аудиофайлам.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, filename))
    return sorted(files)


def _save_transcription_to_txt(
    grouped_segments: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_path: str,
    transcription_date: str,
    duration: float
) -> None:
    """
    Сохраняет транскрибацию и метаданные в текстовый файл.

    :param grouped_segments: Список сгруппированных блоков с текстом.
    :param metadata: Словарь с метаданными файла.
    :param output_path: Полный путь для сохранения результата.
    :param transcription_date: Строка с датой и временем транскрибации.
    :param duration: Общая длительность аудио в секундах.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Блок метаданных (строго в вашем формате)
        f.write("=" * 60 + "\n")
        f.write("МЕТАДАННЫЕ ДОКУМЕНТА\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Имя файла: {metadata['file_name']}\n")
        f.write(f"Формат: {metadata['file_ext']}\n")
        f.write(f"Размер: {_format_file_size(metadata['file_size'])}\n")
        f.write(f"Длительность: {_format_duration(duration)} ({int(duration)} сек.)\n")
        f.write(f"Дата транскрибации: {transcription_date}\n")
        f.write(f"Модель распознавания: Whisper medium (beam_size=1)\n")
        f.write(f"Язык: русский\n")
        f.write("ОПИСАНИЕ:\n")
        f.write("ССЫЛКА НА ИСТОЧНИК:\n\n")
        f.write("=" * 60 + "\n")
        f.write("ТЕКСТ ТРАНСКРИБАЦИИ\n")
        f.write("=" * 60 + "\n\n")

        # Блок текста
        for block in grouped_segments:
            start_str = _format_time(block["interval_start"])
            end_str = _format_time(block["interval_end"])
            f.write(f"[{start_str} - {end_str}] {block['text']}\n")


class AudioTranscriber:
    """
    Фасадный класс для транскрибации аудиофайлов с использованием Whisper.

    Управляет жизненным циклом модели и предоставляет высокоуровневые методы
    для обработки одиночных файлов и целых директорий.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        output_dir: str = ".output",
        interval_seconds: int = 60,
        language: str = "ru",
        cpu_threads: int = 8
    ):
        """
        Инициализирует транскрайбер и загружает модель.

        :param model_size: Размер модели Whisper ('tiny', 'base', 'small', 'medium', 'large').
        :param device: Устройство для вычислений ('cpu' или 'cuda').
        :param compute_type: Тип вычислений ('int8', 'float16', 'float32').
        :param output_dir: Базовая директория для сохранения результатов.
        :param interval_seconds: Интервал группировки текста в секундах.
        :param language: Код языка для распознавания.
        :param cpu_threads: Количество потоков CPU.
        """
        self.output_dir = os.path.abspath(output_dir)
        self.interval_seconds = interval_seconds
        self.language = language

        Logger.info(f"Загрузка модели Whisper '{model_size}' на {device}...", name="AudioTranscriber")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )
        Logger.info("Модель успешно загружена.", name="AudioTranscriber")

    def transcribe_file(self, file_path: str) -> str:
        """
        Транскрибирует один аудиофайл.

        :param file_path: Абсолютный или относительный путь к аудиофайлу.
        :returns: Абсолютный путь к созданному текстовому файлу с транскрибацией.
        :raises FileNotFoundError: Если файл не существует.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        Logger.info(f"Начало обработки: {os.path.basename(file_path)}", name="AudioTranscriber")
        start_time = time.time()

        metadata = _get_file_metadata(file_path)
        transcription_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        waveform, duration = _load_and_resample_audio(file_path)
        Logger.info(f"Аудио загружено. Длительность: {_format_duration(duration)}", name="AudioTranscriber")

        Logger.debug("Запуск транскрибации...", name="AudioTranscriber")
        segments = _transcribe_waveform(waveform, self.model, duration, self.language)

        Logger.debug(f"Группировка по интервалам {self.interval_seconds} сек...", name="AudioTranscriber")
        grouped_segments = _group_segments(segments, self.interval_seconds)

        base_name = os.path.splitext(metadata['file_name'])[0]
        output_path = os.path.join(self.output_dir, f"{base_name}.txt")

        Logger.debug(f"Сохранение результата в: {output_path}", name="AudioTranscriber")
        _save_transcription_to_txt(grouped_segments, metadata, output_path, transcription_date, duration)

        elapsed = time.time() - start_time
        speed = duration / elapsed if duration > 0 else 0
        Logger.info(f"Готово за {elapsed:.2f} сек. (Скорость: {speed:.2f}x). Получено блоков: {len(grouped_segments)}", name="AudioTranscriber")

        return output_path

    def transcribe_directory(self, dir_path: str) -> List[str]:
        """
        Транскрибирует все поддерживаемые аудиофайлы в указанной директории.

        :param dir_path: Путь к директории с аудиофайлами.
        :returns: Список абсолютных путей к созданным текстовым файлам.
        :raises NotADirectoryError: Если путь не является директорией.
        """
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Директория не найдена: {dir_path}")

        Logger.info(f"Поиск файлов в: {dir_path}", name="AudioTranscriber")
        audio_files = _find_audio_files(dir_path)

        if not audio_files:
            Logger.warning("Аудиофайлы не найдены.", name="AudioTranscriber")
            return []

        Logger.info(f"Найдено файлов: {len(audio_files)}", name="AudioTranscriber")

        output_paths = []
        for file_path in audio_files:
            try:
                out_path = self.transcribe_file(file_path)
                output_paths.append(out_path)
            except Exception as e:
                Logger.error(f"Ошибка обработки {file_path}: {e}", name="AudioTranscriber")

        Logger.info(f"Всего обработано файлов: {len(output_paths)} из {len(audio_files)}", name="AudioTranscriber")
        return output_paths


if __name__ == "__main__":
    # Инициализация логгера для демонстрационного запуска
    Logger.init(script_name="transcriber_demo")

    transcriber = AudioTranscriber(
        model_size="medium",
        device="cpu",
        output_dir=".output",
        interval_seconds=60
    )

    # Транскрибация одного файла:
    # result_path = transcriber.transcribe_file(r"C:\path\to\audio.mp3")
    # Logger.info(f"Результат сохранен: {result_path}", name="main")

    # Транскрибация всей папки:
    result_paths = transcriber.transcribe_directory(r"c:\Users\ilya\Downloads\video")
    Logger.info(f"Обработано файлов: {len(result_paths)}", name="main")