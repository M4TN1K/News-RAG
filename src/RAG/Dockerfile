
FROM python:3.10-slim


WORKDIR /app


RUN mkdir -p /app/data
RUN mkdir -p /app/.cache/huggingface

COPY requirements.txt .


# Установка необходимых системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt


COPY . .


ENV DATA_DIR=/app/data
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_ENABLE_OFFLINE=1


EXPOSE 8000


CMD ["uvicorn", "rag_bot_api.main:app", "--host", "0.0.0.0", "--port", "8000"]