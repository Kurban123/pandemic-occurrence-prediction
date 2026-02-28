import os
import random
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

def setup_logger(log_file: Path) -> logging.Logger:
    """Настройка логгера для записи результатов и ошибок."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("PandemicModel")
    logger.setLevel(logging.INFO)
    
    # Предотвращение дублирования логов при повторных запусках
    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file, mode='w')
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger

def enforce_reproducibility(seed: int = 42) -> None:
    """Установка всех random seeds для хардкорной воспроизводимости (требование Nature)."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # Для более старых версий TF, где эта функция отсутствует
        pass