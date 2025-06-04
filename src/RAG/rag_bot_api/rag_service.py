# rag_service.py

from faiss_search.chunk_searcher import ChunkSearcher
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import os

# Импорты из LangChain
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

# Импорт мультиагентской системы
from rag_bot_api.rag_agents import RAGAgentOrchestrator

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, 
                 api_key: str = None, 
                 model_name: str = 'gpt-3.5-turbo',
                 embedding_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 use_agents: bool = True):

        self.searcher = ChunkSearcher()
        self.model = SentenceTransformer(embedding_model_name)
        self.api_key = api_key
        self.use_agents = use_agents

        self.proxy_api_url = os.getenv("PROXY_API_URL", "https://api.proxyapi.ru/openai/v1/chat/completions")

        if api_key:
            self.llm = ChatOpenAI(
                temperature=0,
                model_name=model_name,
                api_key=api_key,
                openai_api_base=self.proxy_api_url.rsplit('/chat/completions', 1)[0]
            )

            # Если указан API ключ и включены агенты, инициализируем оркестратор
            if use_agents:
                self.orchestrator = RAGAgentOrchestrator(
                    llm=self.llm,
                    chunk_searcher=self.searcher,
                    sentence_transformer=self.model,
                    proxy_api_url=self.proxy_api_url
                )
        else:
            logger.warning("API ключ не указан, мультиагентская система будет недоступна")
            self.use_agents = False

    def answer(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Обрабатывает запрос пользователя и возвращает ответ

        Если включен режим мультиагентов и доступен API ключ,
        использует мультиагентскую систему для обработки запроса.
        В противном случае использует базовый подход.
        """

        # Проверяем, можно ли использовать мультиагентскую систему
        if self.use_agents and hasattr(self, 'orchestrator'):
            try:
                # Используем мультиагентскую систему
                logger.info(f"Используем мультиагентскую систему для запроса: {question}")
                result = self.orchestrator.process_query(question, k=k)

                # Добавляем поля для обратной совместимости
                if "context" not in result:
                    result["context"] = "\n\n".join([doc[:200] + "..." for doc in result.get("context_used", [])])

                return result

            except Exception as e:
                logger.error(f"Ошибка при использовании мультиагентской системы: {e}")
                logger.info("Переключаемся на базовый подход")

        logger.info(f"Используем базовый подход для запроса: {question}")
        query_embedding = self.model.encode([question], convert_to_tensor=False)
        results = self.searcher.search(np.array(query_embedding), k=k)

        context = "\n\n".join([result["text"] for result in results])
        return {
            "question": question,
            "context_used": [result["text"][:200] + "..." for result in results],
            "context": context
        }
