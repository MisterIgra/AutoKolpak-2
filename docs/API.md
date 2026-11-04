# API

- `GET /api/health` — состояние конфигурации.
- `GET /api/model-info` — backend и возможности.
- `GET /api/whoami` — версия проекта.
- `POST /api/predict` — multipart `file` и опциональный `reference_outer_diameter_mm` (330–460 мм).

Без Anthropic API key predict возвращает HTTP 503. Миллиметровые оценки возможны только при измеренном reference diameter.