from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
import httpx
import os
import logging
import time
from dotenv import load_dotenv
from rag_bot_api.rag_service import RAGService


load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Получение переменных окружения
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
USE_AGENTS = os.getenv("USE_AGENTS", "True").lower() == "true"
PROXY_API_URL = os.getenv("PROXY_API_URL", "https://api.proxyapi.ru/openai/v1/chat/completions")

app = FastAPI(title="RAG Bot API", description="API для получения информации из новостных источников для Telegram бота")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Запрос {request.url.path} обработан за {process_time:.4f} сек")
    return response


# Инициализация сервиса RAG
rag = RAGService(
    api_key=PROXY_API_KEY,
    model_name=MODEL_NAME,
    use_agents=USE_AGENTS
)

logger.info(f"RAG Bot API запущен. Использование агентов: {USE_AGENTS}, Модель: {MODEL_NAME}, ProxyAPI URL: {PROXY_API_URL}")

prompt_template = """
Ты — новостной блогер. Используй следующую информацию для ответа:

{context}

Вопрос: {question}

Инструкции:
- Опираешься только на эту статью.
- Если информация не соответствует вопросу — скажи об этом.
- Указывай дату и источник, если это релевантно.
"""


class QuestionRequest(BaseModel):
    question: str
    k: int = 3  # Количество документов для поиска (по умолчанию 3)
    detailed: bool = False  # Включить детальную информацию о работе агентов


@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    Обработчик POST-запроса для получения ответа на вопрос от Telegram бота

    Args:
        request: Запрос с вопросом пользователя и дополнительными параметрами

    Returns:
        Ответ на вопрос в формате, подходящем для Telegram бота
    """
    try:
        # Получение контекста с помощью RAG
        rag_result = rag.answer(request.question)

        prompt = prompt_template.format(context=rag_result["context"], question=request.question)

        messages = [
            {"role": "system", "content": "Ты — помощник, который использует предоставленный контекст для ответов."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 200
        }

        headers = {
            "Authorization": f"Bearer {PROXY_API_KEY}",
            "Content-Type": "application/json"
        }

        # Отправка запроса через ProxyAPI
        async with httpx.AsyncClient() as client:
            response = await client.post(PROXY_API_URL, json=payload, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        return {
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e)
            }
        )