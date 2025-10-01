"""
Конфигурация сервиса
"""

import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class Config:
    """Настройки приложения"""
    
    # Сервер
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    
    # Модель
    MODEL_PATH: str = os.getenv("MODEL_PATH", "hypoxia_model.joblib")
    SAMPLING_FREQUENCY: float = float(os.getenv("SAMPLING_FREQUENCY", "4"))
    WINDOW_MINUTES: int = int(os.getenv("WINDOW_MINUTES", "10"))
    MIN_DATA_MINUTES: int = int(os.getenv("MIN_DATA_MINUTES", "5"))
    PREDICTION_INTERVAL: int = int(os.getenv("PREDICTION_INTERVAL", "30"))
    
    # Пороги риска
    RISK_MEDIUM_THRESHOLD: float = float(os.getenv("RISK_MEDIUM_THRESHOLD", "0.3"))
    RISK_HIGH_THRESHOLD: float = float(os.getenv("RISK_HIGH_THRESHOLD", "0.5"))
    RISK_CRITICAL_THRESHOLD: float = float(os.getenv("RISK_CRITICAL_THRESHOLD", "0.8"))
    
    # Логирование
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()

