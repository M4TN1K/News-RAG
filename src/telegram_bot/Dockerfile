### Использовать этот Dockerfile для контейнеризации Telegram-бота на aiogram и связи с RAG API
FROM python:3.12-slim


WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY bot.py .


ENV PYTHONUNBUFFERED=1


CMD ["python", "bot.py"]