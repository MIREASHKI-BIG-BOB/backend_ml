FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY analysis.py .
COPY config.py .
COPY hypoxia_model.joblib .
COPY env.example .env
EXPOSE 8000
CMD ["python", "main.py"]

