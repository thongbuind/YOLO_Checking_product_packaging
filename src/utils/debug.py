# debug.py
# Module debug nâng cao để ghi log CHI TIẾT NHẤT CÓ THỂ, bao gồm:
# - Logging thread-safe với file + console (nếu cần).
# - Ghi exception đầy đủ: time, thread, type, message, traceback, system info, extra context.
# - Redirect TOÀN BỘ stdout/stderr vào log file → "terminal có gì khi crash thì in hết vào".
# - Log thêm: system info (OS, Python version, modules loaded), variables dump nếu pass.
# - Decorator để wrap functions, auto-log input/output/exception.
# - Hỗ trợ log bình thường (info, debug, warning) với context.
#
# Cách dùng trong detect.py:
#   - Import: from debug import setup_logger, log_exception, redirect_output_to_log, debug_wrap
#   - Setup: logger = setup_logger('debug_log.txt')
#   - Redirect output: redirect_output_to_log(logger)  # ← Để ghi hết print() vào file
#   - Wrap functions: @debug_wrap
#     def process_items(self, item_boxes): ...

import logging
import traceback
import datetime
import sys
import os
import platform
import threading
import inspect  # Để lấy stack info chi tiết hơn
from contextlib import contextmanager

class LogOutput:
    """
    Class để redirect stdout/stderr vào logger.
    Mọi print() hoặc sys.stdout.write() sẽ được ghi vào log file với level INFO/ERROR.
    """
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ""  # Buffer để gom dòng

    def write(self, message):
        self._buffer += message
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            for line in lines[:-1]:
                if line.strip():  # Bỏ qua empty lines
                    self.logger.log(self.level, f"[STDOUT/STDERR] {line}")
            self._buffer = lines[-1]

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, f"[STDOUT/STDERR] {self._buffer}")
        self._buffer = ""

def redirect_output_to_log(logger, redirect_stderr=True):
    """
    Redirect sys.stdout và sys.stderr vào logger.
    Gọi 1 lần ở đầu chương trình để "in hết terminal vào file".
    """
    sys.stdout = LogOutput(logger, logging.INFO)
    if redirect_stderr:
        sys.stderr = LogOutput(logger, logging.ERROR)
    logger.info("Redirected stdout/stderr to log file for full terminal capture.")

def setup_logger(log_file='debug_log.txt', level=logging.DEBUG, console_output=False):
    """
    Setup logger chi tiết.
    - log_file: File output.
    - level: DEBUG để ghi hết.
    - console_output: Nếu True, log cả ra console.
    Returns: logger.
    """
    logger = logging.getLogger('assembly_debug')
    logger.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Formatter siêu chi tiết: time, thread, level, file:line, message
    formatter = logging.Formatter(
        '%(asctime)s - Thread:%(threadName)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Thêm handler
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    # Console handler nếu cần
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Log startup với SYSTEM INFO CHI TIẾT
    logger.info("===== LOGGER SETUP COMPLETE =====")
    logger.info(f"Log file: {os.path.abspath(log_file)}")
    logger.info(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working dir: {os.getcwd()}")
    logger.info(f"Loaded modules: {list(sys.modules.keys())}")  # Danh sách modules loaded
    logger.info(f"Environment variables: {dict(os.environ)}")  # Toàn bộ env vars (cẩn thận nếu sensitive)
    logger.info("===== END OF SETUP =====")
    
    return logger

def log_exception(logger, exception, extra_context=None, local_vars=None):
    """
    Ghi exception CHI TIẾT NHẤT:
    - Time, thread, exception type/message.
    - Full traceback với file/line/code.
    - System info nếu cần.
    - Extra context (dict).
    - Dump local_vars nếu pass (e.g., locals() từ nơi gọi).
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb_list = traceback.format_exception(exc_type, exc_value, exc_tb)
    
    logger.error("===== EXCEPTION OCCURRED =====")
    logger.error(f"Time: {now}")
    logger.error(f"Thread: {threading.current_thread().name}")
    logger.error(f"Exception Type: {exc_type.__name__}")
    logger.error(f"Exception Message: {str(exc_value)}")
    
    if extra_context:
        logger.error(f"Extra Context: {extra_context}")
    
    if local_vars:
        logger.error("Local Variables Dump:")
        for var_name, var_value in local_vars.items():
            try:
                logger.error(f"  {var_name}: {repr(var_value)}")  # repr để an toàn
            except:
                logger.error(f"  {var_name}: <unrepr-able>")
    
    logger.error("Full Traceback:")
    for line in tb_list:
        logger.error(line.strip())
    
    # Thêm stack frame chi tiết
    logger.error("Detailed Stack Frames:")
    frame = inspect.currentframe()
    while frame:
        frame_info = inspect.getframeinfo(frame)
        logger.error(f"  File: {frame_info.filename}, Line: {frame_info.lineno}, Function: {frame_info.function}")
        frame = frame.f_back
    
    logger.error("===== END OF EXCEPTION =====")

@contextmanager
def log_context(logger, context_name, extra_info=None):
    """
    Context manager để log enter/exit một block code.
    Ví dụ: with log_context(logger, "Processing Cam1"):
        ...
    """
    logger.debug(f"ENTER: {context_name} {extra_info or ''}")
    try:
        yield
    finally:
        logger.debug(f"EXIT: {context_name}")

def debug_wrap(func):
    """
    Decorator wrap function: log input, output, exception CHI TIẾT.
    Auto dump args/kwargs/locals khi error.
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('assembly_debug')
        func_name = func.__name__
        logger.debug(f"ENTER FUNCTION: {func_name} - Args: {args} - Kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"EXIT FUNCTION: {func_name} - Result: {repr(result)}")
            return result
        except Exception as e:
            # Dump locals() khi error
            local_vars = inspect.currentframe().f_locals
            log_exception(logger, e, extra_context={
                'function': func_name,
                'args': args,
                'kwargs': kwargs
            }, local_vars=local_vars)
            raise  # Re-raise

    return wrapper
