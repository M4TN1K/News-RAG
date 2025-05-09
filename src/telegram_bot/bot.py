import logging
import os
from dotenv import load_dotenv
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.client.bot import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message
import httpx

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = str(os.getenv("TELEGRAM_BOT_TOKEN"))
API_URL = os.getenv("RAG_API_URL", "http://rag-app:8000/ask")
logger.info("RAG API URL = %s", API_URL)

bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher()

async def ask_api(question: str) -> dict:
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(API_URL, json={"question": question})
        resp.raise_for_status()
        return resp.json()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я — RAG-бот. Задайте любой вопрос, "
        "я запрошу контекст и постараюсь ответить."
    )

@dp.message(lambda m: m.text and not m.text.startswith("/"))
async def handle_message(message: Message):
    question = message.text.strip()
    if not question:
        return

    await message.chat.do("typing")

    try:
        result = await ask_api(question)
        answer = result.get("answer", "Извините, не смог найти ответ.")
        await message.answer(answer)
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error: %s", e)
        await message.answer("Сервис временно недоступен.")
    except Exception as e:
        logger.exception("Unexpected error")
        await message.answer(f"Произошла ошибка: {e}")

async def main():
    await bot.delete_webhook(drop_pending_updates=True)

    dp.message.register(cmd_start, Command(commands=["start"]))
    dp.message.register(handle_message, lambda m: m.text and not m.text.startswith("/"))


    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
