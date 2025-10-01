# CTG Hypoxia Prediction API

WebSocket-based real-time fetal hypoxia prediction service.

## Installation

```bash
pip install -r requirements.txt
python main.py
```

Default endpoint: `ws://localhost:8000/ws/ctg`

## WebSocket Protocol

### Endpoint

`ws://localhost:8000/ws/ctg`

### Request Format

```json
{
  "sensorID": "sensor_001",
  "secFromStart": 125.5,
  "data": {
    "BPMChild": 142.5,
    "uterus": 15.3,
    "spasms": 0.0
  }
}
```

### Response Format

```json
{
  "sensorID": "sensor_001",
  "secFromStart": 125.5,
  "data": {
    "BPMChild": 142.5,
    "uterus": 15.3,
    "spasms": 0.0
  },
  "prediction": {
    "hypoxia_pro
    "hypoxia_risk": "low",
    "alerts": [
      "🟡 Низкая вариабельность: 4.2 bpm"
    ],
    "confidence": 0.85,
    "recommendations": [
      "Продолжить рутинный мониторинг"
    ]
  },
  "status": "ok",
  "timestamp": "2025-10-01T12:34:56.789Z"
}
```

### Response Fields

| Field          | Type         | Description                                           |
| -------------- | ------------ | ----------------------------------------------------- |
| `sensorID`     | string       | Sensor identifier (echoed from request)               |
| `secFromStart` | float        | Time in seconds (echoed from request)                 |
| `data`         | object       | Sensor data (echoed from request)                     |
| `prediction`   | object\|null | Prediction result (null if insufficient data)         |
| `status`       | string       | Response status: `ok`, `warning`, `critical`, `error` |
| `timestamp`    | string       | ISO 8601 timestamp                                    |

### Prediction Object

| Field                 | Type     | Description                                     |
| --------------------- | -------- | ----------------------------------------------- |
| `hypoxia_probability` | float    | Hypoxia probability [0.0-1.0]                   |
| `hypoxia_risk`        | string   | Risk level: `low`, `medium`, `high`, `critical` |
| `alerts`              | string[] | Clinical alerts                                 |
| `confidence`          | float    | Model confidence [0.0-1.0]                      |
| `recommendations`     | string[] | Medical recommendations                         |

## Behavior

### Data Accumulation

- Minimum 5 minutes of data required for initial prediction
- Sliding window of 10 minutes used for analysis
- Predictions generated every 30 seconds

### Validation

- Valid FHR range: 80-200 bpm
- Invalid data points are filtered automatically
- Prediction returns null if insufficient valid data

### Risk Levels

| Probability | Level    | Status Code |
| ----------- | -------- | ----------- |
| 0.0 - 0.3   | low      | ok          |
| 0.3 - 0.5   | medium   | ok          |
| 0.5 - 0.8   | high     | warning     |
| 0.8 - 1.0   | critical | critical    |

## HTTP Endpoints

**GET /** - Service health check  
**GET /health** - Detailed health status  
**GET /sensors** - List active sensor sessions  
**POST /reset/{sensor_id}** - Reset analyzer for specific sensor

## Configuration

Required files:

- `hypoxia_model.joblib` - Trained ML model
- `.env` - Environment variables (optional)

Key parameters in `.env`:

```
PORT=8000
MODEL_PATH=hypoxia_model.joblib
WINDOW_MINUTES=10
SAMPLING_FREQUENCY=4
```

## Integration Requirements

### Data Stream

- Sampling rate: 4 Hz (one measurement per 0.25 seconds)
- FHR valid range: 80-200 bpm
- Continuous stream required for accurate predictions

### Session Management

- One WebSocket connection per sensor
- Analyzer state persists during connection
- Automatic cleanup on disconnect

### Go Client Types

```go
type SensorData struct {
    BPMChild float32 `json:"BPMChild"`
    Uterus   float32 `json:"uterus"`
    Spasms   float32 `json:"spasms"`
}

type MessageData struct {
    SensorID     string     `json:"sensorID"`
    SecFromStart float32    `json:"secFromStart"`
    Data         SensorData `json:"data"`
}

type Prediction struct {
    HypoxiaProbability float64  `json:"hypoxia_probability"`
    HypoxiaRisk       string   `json:"hypoxia_risk"`
    Alerts            []string `json:"alerts"`
    Confidence        float64  `json:"confidence"`
    Recommendations   []string `json:"recommendations"`
}

type ResponseData struct {
    SensorID     string      `json:"sensorID"`
    SecFromStart float32     `json:"secFromStart"`
    Data         SensorData  `json:"data"`
    Prediction   *Prediction `json:"prediction"`
    Status       string      `json:"status"`
    Timestamp    string      `json:"timestamp"`
}
```

## Error Handling

| Error        | Status  | Cause                         |
| ------------ | ------- | ----------------------------- |
| Invalid JSON | error   | Malformed request             |
| Model error  | error   | Prediction failure            |
| No valid FHR | warning | All values outside 80-200 bpm |

## Constraints

- Minimum 5 minutes data accumulation before first prediction
- Predictions throttled to 30-second intervals
- Maximum buffer size: 2x window size (20 minutes at 4 Hz)
- Per-sensor isolation (no cross-contamination)
