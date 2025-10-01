"""
FastAPI WebSocket сервис для real-time предсказания гипоксии плода по данным КТГ.
"""

import asyncio
import json
import logging
from typing import Dict, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analysis import LiveCTGAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(
    title="CTG Hypoxia Prediction Service",
    description="Real-time fetal hypoxia prediction via WebSocket",
    version="1.0.0"
)

# CORS для фронтенда (если нужно)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище активных анализаторов для каждого сенсора
active_analyzers: Dict[str, LiveCTGAnalyzer] = {}


# ============= Модели данных =============

class SensorData(BaseModel):
    """Данные с датчиков КТГ"""
    BPMChild: float  # ЧСС плода
    uterus: float    # Тонус матки
    spasms: float    # Спазмы (не используется в текущей модели)


class MessageData(BaseModel):
    """Входящее сообщение от Go-бэкенда"""
    sensorID: str
    secFromStart: float
    data: SensorData


class PredictionResult(BaseModel):
    """Результат предсказания"""
    hypoxia_probability: float  # Вероятность гипоксии (0-1)
    hypoxia_risk: str          # "low", "medium", "high", "critical"
    alerts: list[str]          # Список предупреждений
    confidence: float          # Уверенность модели
    recommendations: list[str] # Рекомендации врачу


class ResponseData(BaseModel):
    """Ответное сообщение в Go-бэкенд"""
    sensorID: str
    secFromStart: float
    data: SensorData
    prediction: Optional[PredictionResult] = None
    status: str = "ok"  # ok, warning, error
    timestamp: str


# ============= WebSocket Endpoint =============

@app.websocket("/ws/ctg")
async def websocket_ctg_analysis(websocket: WebSocket):
    """
    WebSocket endpoint для приема данных КТГ и отправки предсказаний.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    current_sensor_id: Optional[str] = None
    
    try:
        while True:
            # Получаем данные от Go-бэкенда
            raw_data = await websocket.receive_text()
            
            try:
                # Парсим JSON
                message_dict = json.loads(raw_data)
                message = MessageData(**message_dict)
                
                sensor_id = message.sensorID
                current_sensor_id = sensor_id
                
                # Создаем или получаем анализатор для этого сенсора
                if sensor_id not in active_analyzers:
                    logger.info(f"Creating new analyzer for sensor {sensor_id}")
                    active_analyzers[sensor_id] = LiveCTGAnalyzer(
                        model_path="hypoxia_model.joblib",
                        fs=4,  # 4 Гц по умолчанию
                        window_minutes=10,
                        alert_buffer_sec=30,
                        use_ml_prediction=True
                    )
                
                analyzer = active_analyzers[sensor_id]
                
                # Добавляем данные в анализатор
                analyzer.add_data(
                    timestamp=message.secFromStart,
                    fhr=message.data.BPMChild,
                    uc=message.data.uterus
                )
                
                # Получаем предсказание
                prediction_data = analyzer.get_prediction()
                
                # Формируем ответ
                if prediction_data:
                    prediction = PredictionResult(
                        hypoxia_probability=prediction_data["hypoxia_probability"],
                        hypoxia_risk=prediction_data["risk_level"],
                        alerts=prediction_data["alerts"],
                        confidence=prediction_data["confidence"],
                        recommendations=prediction_data["recommendations"]
                    )
                    
                    status = "warning" if prediction_data["hypoxia_probability"] > 0.5 else "ok"
                    
                    if prediction_data["hypoxia_probability"] > 0.8:
                        status = "critical"
                else:
                    # Недостаточно данных для предсказания
                    prediction = None
                    status = "ok"
                
                # Отправляем ответ обратно
                response = ResponseData(
                    sensorID=sensor_id,
                    secFromStart=message.secFromStart,
                    data=message.data,
                    prediction=prediction,
                    status=status,
                    timestamp=datetime.utcnow().isoformat()
                )
                
                await websocket.send_text(response.model_dump_json())
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_response = {
                    "status": "error",
                    "message": f"Invalid JSON format: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(error_response))
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                error_response = {
                    "status": "error",
                    "message": f"Processing error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for sensor {current_sensor_id}")
        # Очищаем анализатор при отключении
        if current_sensor_id and current_sensor_id in active_analyzers:
            del active_analyzers[current_sensor_id]
    
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket: {e}", exc_info=True)
        if current_sensor_id and current_sensor_id in active_analyzers:
            del active_analyzers[current_sensor_id]


# ============= HTTP Endpoints =============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "CTG Hypoxia Prediction",
        "status": "online",
        "version": "1.0.0",
        "active_sensors": len(active_analyzers)
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "active_analyzers": len(active_analyzers),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/reset/{sensor_id}")
async def reset_sensor(sensor_id: str):
    """Сброс анализатора для конкретного сенсора"""
    if sensor_id in active_analyzers:
        del active_analyzers[sensor_id]
        return {"status": "reset", "sensor_id": sensor_id}
    return {"status": "not_found", "sensor_id": sensor_id}


@app.get("/sensors")
async def list_active_sensors():
    """Список активных сенсоров"""
    sensors_info = {}
    for sensor_id, analyzer in active_analyzers.items():
        sensors_info[sensor_id] = {
            "data_points": len(analyzer.data),
            "has_prediction": analyzer.last_prediction is not None
        }
    return sensors_info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )

