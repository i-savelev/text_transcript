import logging
import os
import queue
import datetime
from typing import Optional
import inspect


class NonLockingFileHandler(logging.FileHandler):
    """
    FileHandler который открывает файл только на время записи.
    После каждой записи файл закрывается, что позволяет удалять/перемещать
    файл лога без ошибок блокировки.
    """

    def emit(self, record):
        """Записать лог-сообщение, открывая и закрывая файл для каждой записи."""
        if self.stream is None:
            self.stream = self._open()
        try:
            logging.StreamHandler.emit(self, record)
            self.flush()
        finally:
            self.close()


class Logger:
    """
    Фасад для удобного статического логирования без необходимости создания экземпляров.
    Использует модуль `logging` и автоматически инициализирует файловый обработчик
    при первом вызове любого метода логирования, если не был инициализирован ранее.

    Поддерживает подключение GUI-очереди через `attach_gui_queue()` — все сообщения
    будут дублироваться в очередь для отображения в интерфейсе приложения.

    Example:
        >>> Logger.info("Приложение запущено")
        >>> Logger.error("Произошла ошибка", name="db")
    """

    _initialized: bool = False
    _log_file: str = ".log/app.log"
    _level: int = logging.DEBUG
    _logger: Optional[logging.Logger] = None
    _gui_queue: Optional[queue.Queue] = None  # Очередь для GUI

    @staticmethod
    def get_temp_path() -> str:
        """
        Возвращает путь к файлу app.log в системной временной папке.
        Использует переменные окружения TMP, TEMP, TMPDIR.
        Если не найдены — использует:
        - '/tmp' на Unix-системах
        - 'AppData\\Local\\Temp' на Windows
        """
        tmp_dir = '/tmp' if os.name != 'nt' else os.path.expanduser('~/AppData/Local/Temp')
        return tmp_dir

    @classmethod
    def attach_gui_queue(cls, gui_queue: queue.Queue) -> None:
        """
        Подключает очередь GUI для дублирования логов в интерфейс.

        Все сообщения, записанные через Logger (включая raw_dataframe и separator),
        будут дополнительно отправляться в эту очередь.

        :param gui_queue: Очередь queue.Queue, которую читает GUI-панель лога.
        """
        cls._ensure_initialized()
        cls._gui_queue = gui_queue

    @classmethod
    def _send_to_gui(cls, text: str) -> None:
        """
        Отправляет текст в GUI-очередь, если она подключена.
        Безопасный метод — не делает ничего, если очередь не подключена.

        :param text: Текст для отправки в панель лога.
        """
        if cls._gui_queue is not None:
            cls._gui_queue.put(text)

    @classmethod
    def _ensure_initialized(cls) -> None:
        """
        Гарантирует, что логгер инициализирован ровно один раз.
        Создаёт директорию для логов, настраивает форматтер и файловый обработчик,
        а также отключает распространение логов выше по иерархии (чтобы избежать дублирования).

        Не-файловые обработчики (например, QueueHandler для GUI) сохраняются.

        :raises OSError: Если не удаётся создать директорию для логов.
        """
        if cls._initialized:
            return

        cls._log_file = os.path.abspath(cls._log_file)
        log_dir = os.path.dirname(cls._log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # NonLockingFileHandler закрывает файл после каждой записи
        file_handler = NonLockingFileHandler(cls._log_file, encoding='utf-8', delay=True)
        file_handler.setLevel(cls._level)
        file_handler.setFormatter(formatter)

        cls._logger = logging.getLogger("log")

        # Удаляем ТОЛЬКО файловые обработчики, сохраняя остальные (QueueHandler для GUI)
        for handler in cls._logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                cls._logger.removeHandler(handler)

        cls._logger.setLevel(cls._level)
        cls._logger.addHandler(file_handler)
        cls._logger.propagate = False  # избегаем дублирования

        cls._initialized = True

    @classmethod
    def configure(cls, log_file: str = ".log/app.log", level: int = logging.INFO) -> None:
        """
        Настраивает параметры логирования до первого использования.
        Сбрасывает флаг инициализации, чтобы при следующем вызове логирования
        применить новые настройки.

        :param log_file: Абсолютный или относительный путь к файлу лога.
        :param level: Уровень логирования (например, `logging.DEBUG`, `logging.INFO`).
        :raises ValueError: Если `log_file` — пустая строка или не строка.
        :raises TypeError: Если `level` не является целым числом.
        """
        if not isinstance(log_file, str) or not log_file.strip():
            raise ValueError("log_file must be a non-empty string.")
        if not isinstance(level, int):
            raise TypeError("level must be an integer.")

        cls._log_file = os.path.abspath(log_file)
        cls._level = level
        cls._initialized = False  # сброс для переинициализации

    @classmethod
    def init(cls, script_name: str) -> None:
        """
        Инициализирует логгер и записывает заголовок в начало лог-файла.
        Заголовок содержит имя скрипта и текущее время запуска.
        Если файл уже существует — он очищается.

        Не-файловые обработчики (QueueHandler для GUI) сохраняются.

        :param script_name: Название скрипта для отображения в логе.
        :raises ValueError: Если script_name — пустая строка.
        :raises OSError: Если не удаётся записать в файл.
        """
        if not script_name or not isinstance(script_name, str):
            raise ValueError("script_name must be a non-empty string.")

        # Гарантируем, что путь инициализирован
        cls._ensure_initialized()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (
            f"{'='*50}\n"
            f"СКРИПТ: {script_name}\n"
            f"ЗАПУСК: {timestamp}\n"
            f"{'='*50}\n"
        )

        # Закрываем ТОЛЬКО файловые обработчики, чтобы можно было перезаписать файл.
        # Не-файловые обработчики (QueueHandler для GUI) не трогаем.
        if cls._logger is not None:
            for handler in cls._logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    cls._logger.removeHandler(handler)
            cls._initialized = False  # Перезапустим инициализацию позже

        # Записываем заголовок в файл (очистка файла)
        log_dir = os.path.dirname(cls._log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(cls._log_file, 'w', encoding='utf-8') as f:
            f.write(header)

        # Перезапускаем инициализацию, чтобы восстановить файловый обработчик
        cls._ensure_initialized()

        # Отправляем заголовок в GUI-панель
        cls._send_to_gui(header.rstrip('\n'))

    @classmethod
    def separator(cls, sep: str = '-'):
        """Записывает разделительную линию в лог-файл и GUI."""
        cls._ensure_initialized()
        line = f'{sep*80}'
        with open(cls._log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        cls._send_to_gui(line)

    @classmethod
    def _get_logger(cls, name: str) -> logging.Logger:
        """
        Возвращает дочерний логгер с заданным именем.
        Гарантирует инициализацию основного логгера при необходимости.

        :param name: Имя дочернего логгера.
        :return: Экземпляр `logging.Logger`.
        """
        cls._ensure_initialized()
        return cls._logger.getChild(name)

    @classmethod
    def debug(cls, message: str = '', name: str = '') -> None:
        """Записывает сообщение уровня DEBUG."""
        cls._get_logger(name).debug(message, stacklevel=2)

    @classmethod
    def info(cls, message: str = '', name: str = '') -> None:
        """Записывает сообщение уровня INFO."""
        cls._ensure_initialized()
        cls._get_logger(name).info(message, stacklevel=2)

    @classmethod
    def warning(cls, message: str = '', name: str = '') -> None:
        """Записывает сообщение уровня WARNING."""
        cls._ensure_initialized()
        cls._get_logger(name).warning(message, stacklevel=2)

    @classmethod
    def error(cls, message: str = '', name: str = '') -> None:
        """Записывает сообщение уровня ERROR."""
        cls._ensure_initialized()
        cls._get_logger(name).error(message, stacklevel=2)

    @classmethod
    def critical(cls, message: str = '', name: str = '') -> None:
        """Записывает сообщение уровня CRITICAL."""
        cls._ensure_initialized()
        cls._get_logger(name).critical(message, stacklevel=2)

    @classmethod
    def path(cls):
        """Возвращает путь к текущему лог-файлу."""
        return cls._log_file

    @classmethod
    def data(cls, data: list | tuple | dict, label: str = '', name: str = 'data.', max_items=20):
        """
        Логировать структуру данных для отладки.

        :param name: Имя модуля
        :param label: Описание данных
        :param data: Данные (dict, list, или любой объект)
        :param max_items: Максимум элементов для вывода
        """
        cls._ensure_initialized()
        cls._get_logger(name).debug(msg=f"ДАННЫЕ [{label}]:", stacklevel=2)

        if isinstance(data, dict):
            cls._get_logger(name).debug(msg=f"Тип: dict, Кол-во: {len(data)}", stacklevel=2)
            for i, (k, v) in enumerate(data.items()):
                if i >= max_items:
                    cls._get_logger(name).debug(msg=f"... и ещё {len(data) - max_items}", stacklevel=2)
                    break
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:50] + "..."
                cls._get_logger(name).debug(msg=f"   [{k}] = {v_str}", stacklevel=2)

        elif isinstance(data, (list, tuple)):
            cls._get_logger(name).debug(msg=f"Тип: {type(data).__name__}, Кол-во: {len(data)}", stacklevel=2)
            for i, item in enumerate(data):
                if i >= max_items:
                    cls._get_logger(name).debug(msg=f"... и ещё {len(data) - max_items}", stacklevel=2)
                    break
                item_str = str(item)
                if len(item_str) > 50:
                    item_str = item_str[:50] + "..."
                cls._get_logger(name).debug(msg=f"   [{i}] {item_str}", stacklevel=2)
        else:
            cls._get_logger(name).debug(msg=f"Тип: {type(data).__name__}", stacklevel=2)
            cls._get_logger(name).debug(msg=f"  Значение: {data}", stacklevel=2)

    @classmethod
    def raw_dataframe(cls, df, caption: str = "", max_rows: int = 50, max_cols: int = None) -> None:
        """
        Записывает pandas DataFrame в лог-файл «как есть» — без временных меток и форматтера.
        Также дублирует вывод в GUI-очередь, если она подключена.

        :param df: pandas DataFrame
        :param caption: Опциональная подпись над таблицей
        :param max_rows: Максимум строк для вывода
        :param max_cols: Максимум столбцов для вывода (None = все)
        """
        try:
            import pandas as pd
        except ImportError:
            cls.error("Pandas не установлен, невозможно записать DataFrame", name="logger")
            return

        if not isinstance(df, pd.DataFrame):
            cls.error(f"Объект типа {type(df).__name__} не является DataFrame", name="logger")
            return

        cls._ensure_initialized()

        # Формируем строковое представление таблицы
        df_display = df.head(max_rows)
        df_str = df_display.to_string(
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=False,
            line_width=180,
            index=True
        )

        # Подготовка содержимого для записи
        lines = []
        if caption:
            lines.append(f"ТАБЛИЦА: {caption}")
            lines.append(f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
            lines.append('=' * 80)
        else:
            lines.append("")  # пустая строка перед таблицей

        lines.extend(df_str.split('\n'))

        if len(df) > max_rows:
            lines.append(f"\n... (показаны первые {max_rows} из {len(df)} строк)")

        lines.append('=' * 80)
        lines.append("")  # пустая строка после таблицы

        # Записываем напрямую в файл — без форматтера и временных меток
        try:
            with open(cls._log_file, 'a', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
        except OSError as e:
            cls.error(f"Не удалось записать таблицу в лог: {e}", name="logger")

        # Дублируем в GUI-очередь
        for line in lines:
            cls._send_to_gui(line)


if __name__ == '__main__':
    def run():
        Logger.info('as', "Тест логирования")

    run()