# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY main.py .
COPY analysis.py .
COPY config.py .

# Копируем модель
COPY hypoxia_model.joblib .

# Копируем файл с переменными окружения (опционально, для дефолтных значений)
COPY env.example .env

# Настраиваем переменные окружения по умолчанию
# Эти значения будут использоваться, если не переопределены при запуске контейнера
ENV HOST=0.0.0.0
ENV PORT=8081
ENV RELOAD=false
ENV MODEL_PATH=hypoxia_model.joblib
ENV SAMPLING_FREQUENCY=4
ENV WINDOW_MINUTES=10
ENV MIN_DATA_MINUTES=5
ENV PREDICTION_INTERVAL=30
ENV RISK_MEDIUM_THRESHOLD=0.3
ENV RISK_HIGH_THRESHOLD=0.5
ENV RISK_CRITICAL_THRESHOLD=0.8
ENV LOG_LEVEL=INFO

# Открываем порт (используем переменную окружения)
EXPOSE ${PORT}

# Команда запуска приложения
CMD ["python", "main.py"]

