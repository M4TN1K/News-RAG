# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_bot_api.rag_service import RAGService
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv



app = FastAPI(title="RAG Bot API")

rag = RAGService()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_API_TOKEN")
)

model_name = "HuggingFaceH4/zephyr-7b-beta"

prompt_template  = """
Ты помощник, который отвечает на вопросы, используя предоставленный контекст.

Контекст:
{context}

Вопрос:
{question}
"""


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    Обработчик POST-запроса для получения ответа на вопрос.

    :param request: Объект запроса, содержащий вопрос.
    :return: Словар с ответом и использованным контекстом.
    """
    try:
        rag_result = rag.answer(request.question)

        prompt = prompt_template.format(context=rag_result["context"], question=request.question)

        messages = [
            {"role": "system", "content": "Ты — помощник, который использует предоставленный контекст для ответов."},
            {"role": "user", "content": prompt}
        ]

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=200
        )

        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "context_used": rag_result["context_used"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))