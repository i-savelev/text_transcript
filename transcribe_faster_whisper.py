import os
import time
import av
import numpy as np
from datetime import datetime
from tqdm import tqdm
from faster_whisper import WhisperModel
from collections import defaultdict

# Отключаем предупреждения Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ============================================
# НАСТРОЙКИ
# ============================================
INPUT_FILE = r"C:\Users\ilya\Downloads\25.09 Консультационный семинар по применению IFC на этапе АГР (№14) _ Ответы на вопросы_mp3.mp3"

# Интервал группировки в секундах
# 30  — компактно, навигация точная
# 60  — очень компактно (по 1 минуте)  <-- РЕКОМЕНДУЮ ДЛЯ ИИ
# 300 — супер-компактно (по 5 минут)
INTERVAL_SECONDS = 60
# ============================================

def format_time(seconds):
    """Конвертирует секунды в формат ЧЧ:ММ:СС или ММ:СС"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def format_duration(seconds):
    """Форматирует длительность в читаемый вид: 1ч 23мин 45с"""
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

def format_file_size(size_bytes):
    """Форматирует размер файла в читаемый вид"""
    for unit in ['Б', 'КБ', 'МБ', 'ГБ']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} ТБ"

def load_and_resample_audio(file_path: str):
    """Надежно читает аудио и конвертирует в 16kHz mono float32 для Whisper"""
    print(f"🎯 Чтение и подготовка аудио: {os.path.basename(file_path)}")
    
    container = av.open(file_path)
    stream = container.streams.audio[0]
    
    resampler = av.audio.resampler.AudioResampler(
        format="flt",
        layout="mono",
        rate=16000
    )
    
    audio_data = []
    for frame in container.decode(stream):
        frame.pts = None
        for resampled_frame in resampler.resample(frame):
            audio_data.append(resampled_frame.to_ndarray().flatten())
            
    waveform = np.concatenate(audio_data)
    duration = len(waveform) / 16000.0
    
    print(f"✅ Аудио загружено. Длительность: {duration:.1f} сек. ({format_duration(duration)})\n")
    return waveform, duration

def get_file_metadata(file_path: str):
    """Собирает метаданные о файле"""
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_name, file_size, file_ext

def write_metadata(f, file_name, file_size, file_ext, duration, transcription_date):
    """Записывает блок метаданных в начало файла"""
    f.write("=" * 60 + "\n")
    f.write("МЕТАДАННЫЕ ДОКУМЕНТА\n")
    f.write("=" * 60 + "\n\n")
    
    # Автоматически заполняемые поля
    f.write(f"Имя файла: {file_name}\n")
    f.write(f"Формат: {file_ext}\n")
    f.write(f"Размер: {format_file_size(file_size)}\n")
    f.write(f"Длительность: {format_duration(duration)} ({int(duration)} сек.)\n")
    f.write(f"Дата транскрибации: {transcription_date}\n")
    f.write(f"Модель распознавания: Whisper medium (beam_size=1)\n")
    f.write(f"Язык: русский\n\n")
    
    # Пустые поля для заполнения пользователем
    f.write("-" * 60 + "\n")
    f.write("ОПИСАНИЕ (заполнить вручную):\n")
    f.write("-" * 60 + "\n")
    f.write("[Описание содержания, темы, участников, контекста]\n\n")
    
    f.write("-" * 60 + "\n")
    f.write("ССЫЛКА НА ИСТОЧНИК (заполнить вручную):\n")
    f.write("-" * 60 + "\n")
    f.write("[URL, путь к файлу, или другая информация об источнике]\n\n")
    
    f.write("-" * 60 + "\n")
    f.write("КЛЮЧЕВЫЕ СЛОВА / ТЕГИ (заполнить вручную):\n")
    f.write("-" * 60 + "\n")
    f.write("[тег1, тег2, тег3]\n\n")
    
    f.write("=" * 60 + "\n")
    f.write("ТЕКСТ ТРАНСКРИБАЦИИ\n")
    f.write("=" * 60 + "\n\n")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл не найден: {INPUT_FILE}")
        return

    base_name = os.path.splitext(INPUT_FILE)[0]
    output_file = f"{base_name}.txt"

    # 1. Сбор метаданных файла
    file_name, file_size, file_ext = get_file_metadata(INPUT_FILE)
    transcription_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. Подготовка аудио
    waveform, duration = load_and_resample_audio(INPUT_FILE)

    # 3. Загрузка модели medium с оптимизацией под CPU
    print("⏳ Загрузка модели Whisper 'medium'...")
    model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8",
        cpu_threads=8,  # Укажите количество физических ядер вашего процессора
    )
    
    print(f"🚀 Начало расшифровки...")
    start_time = time.time()
    
    segments, info = model.transcribe(
        waveform, 
        language="ru", 
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
    )
    
    # 4. Группировка сегментов по интервалам
    intervals = defaultdict(lambda: {"start": None, "end": None, "text_parts": []})
    
    with tqdm(total=duration, unit="сек", desc="Обработка") as pbar:
        for segment in segments:
            interval_idx = int(segment.start // INTERVAL_SECONDS)
            bucket = intervals[interval_idx]
            
            if bucket["start"] is None:
                bucket["start"] = segment.start
            bucket["end"] = segment.end
            bucket["text_parts"].append(segment.text.strip())
            
            pbar.update(segment.end - pbar.n)

    # 5. Запись файла с метаданными
    with open(output_file, "w", encoding="utf-8") as f:
        # Сначала метаданные
        write_metadata(f, file_name, file_size, file_ext, duration, transcription_date)
        
        # Затем текст с таймкодами
        for idx in sorted(intervals.keys()):
            bucket = intervals[idx]
            start = format_time(bucket["start"])
            end = format_time(bucket["end"])
            
            text = " ".join(bucket["text_parts"])
            text = " ".join(text.split())
            
            f.write(f"[{start} - {end}] {text}\n")

    elapsed_time = time.time() - start_time
    
    print(f"\n✅ Готово! Текст сохранен в: {output_file}")
    print(f"⏱️ Затраченное время: {elapsed_time:.2f} сек. ({format_duration(elapsed_time)})")
    print(f"📦 Интервал группировки: {INTERVAL_SECONDS} сек.")
    print(f"📊 Получено блоков: {len(intervals)}")
    
    if duration > 0:
        speed = duration / elapsed_time
        print(f"⚡ Скорость обработки: {speed:.2f}x от реальной длительности")

if __name__ == "__main__":
    main()