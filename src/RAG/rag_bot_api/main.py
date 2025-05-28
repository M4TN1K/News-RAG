from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from rag_bot_api.rag_service import RAGService


load_dotenv()

app = FastAPI(title="RAG Bot API")


rag = RAGService()


PROXY_API_KEY = os.getenv("PROXY_API_KEY")
MODEL_NAME = "gpt-3.5-turbo"


PROXY_API_URL = "https://api.proxyapi.ru/openai/v1/chat/completions"

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


@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    Обработчик POST-запроса для получения ответа
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
            "answer": answer,
            "context_used": rag_result["context_used"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))