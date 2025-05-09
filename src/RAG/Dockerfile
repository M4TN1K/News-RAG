
FROM python:3.10-slim


WORKDIR /app


RUN mkdir -p /app/data

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


COPY . .


ENV DATA_DIR=/app/data


EXPOSE 8000


CMD ["uvicorn", "rag_bot_api.main:app", "--host", "0.0.0.0", "--port", "8000"]