# AutoKolpak AI v2

Система анализа автомобильных колпаков: dataset validation, multitask ML-ядро, параметрический экспорт и русскоязычный FastAPI frontend.

## Запуск

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-web.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Для API создайте `.env` по `.env.example` и задайте Anthropic key. Без ключа health работает, а predict честно возвращает 503. Диаметр reference разрешён только 330–460 мм. Не размещайте секреты в Git.