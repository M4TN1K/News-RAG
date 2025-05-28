# rag_agents.py

import logging
import numpy as np
import os
import httpx
import json
from typing import List, Dict, Any, Optional

# Импорты из LangChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Агент для анализа запроса"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._setup_chain()

    def _setup_chain(self):
        template = """
        Ты эксперт по анализу запросов пользователей о новостях.
        Твоя задача - проанализировать запрос и определить:
        1. Основное намерение пользователя
        2. Ключевые термины и концепции
        3. Потенциальные направления поиска

        Запрос пользователя: {query}

        Анализ запроса:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["query"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_key="query_analysis"
        )

    def analyze(self, query: str) -> str:
        """Анализирует запрос пользователя"""
        try:
            result = self.chain({"query": query})
            return result["query_analysis"]
        except Exception as e:
            logger.error(f"Ошибка при анализе запроса: {e}")
            return ""


class QueryExpander:
    """Агент для расширения запроса"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._setup_chain()

    def _setup_chain(self):
        template = """
        На основе анализа запроса пользователя, расширь его для улучшения поиска.
        Создай несколько альтернативных формулировок, которые помогут найти релевантные новости.

        Исходный запрос: {query}
        Анализ запроса: {query_analysis}

        Расширенный запрос (в формате: основной запрос, альтернативная формулировка 1, альтернативная формулировка 2):
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "query_analysis"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_key="expanded_query"
        )

    def expand(self, query: str, query_analysis: str) -> str:
        """Расширяет запрос на основе анализа"""
        try:
            result = self.chain({
                "query": query, 
                "query_analysis": query_analysis
            })
            return result["expanded_query"]
        except Exception as e:
            logger.error(f"Ошибка при расширении запроса: {e}")
            return query


class DocumentReranker:
    """Агент для переранжирования документов"""

    def __init__(self, api_key: Optional[str] = None, top_k: int = 5):
        self.api_key = api_key
        self.top_k = top_k
        self._setup_reranker()

    def _setup_reranker(self):
        try:
            self.reranker = CohereRerank(
                cohere_api_key=self.api_key,
                top_n=self.top_k
            )
        except Exception as e:
            logger.error(f"Ошибка при инициализации Cohere Reranker: {e}")
            self.reranker = None

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Переранжирует документы для повышения релевантности"""
        if not self.reranker:
            logger.warning("Reranker не инициализирован, возвращаем исходные документы")
            return documents[:self.top_k]

        try:
            # Конвертируем в формат LangChain Document
            langchain_docs = []
            for doc in documents:
                langchain_docs.append(Document(
                    page_content=doc["text"],
                    metadata={
                        "score": doc.get("score", 0),
                        "created_at": doc.get("created_at", ""),
                        "article_id": doc.get("article_id", 0),
                        "metadata": doc.get("metadata", {})
                    }
                ))

            # Выполняем переранжирование
            reranked_docs = self.reranker.compress_documents(langchain_docs, query=query)

            # Конвертируем обратно в исходный формат
            result = []
            for doc in reranked_docs:
                result.append({
                    "text": doc.page_content,
                    "score": doc.metadata.get("score", 0),
                    "created_at": doc.metadata.get("created_at", ""),
                    "article_id": doc.metadata.get("article_id", 0),
                    "metadata": doc.metadata.get("metadata", {})
                })

            return result

        except Exception as e:
            logger.error(f"Ошибка при переранжировании документов: {e}")
            return documents[:self.top_k]


class AnswerGenerator:
    """Агент для генерации ответа на основе контекста"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._setup_chain()

    def _setup_chain(self):
        template = """
        Ты - новостной ассистент. Используй предоставленный контекст для формирования ответа на вопрос пользователя.

        Контекст:
        {context}

        Вопрос: {query}

        Инструкции:
        - Опирайся только на информацию из контекста
        - Если информации недостаточно, укажи на это
        - Будь точным и информативным
        - Указывай даты и источники, если это релевантно

        Ответ:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "context"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_key="answer"
        )

    def generate(self, query: str, context: str) -> str:
        """Генерирует ответ на основе контекста"""
        try:
            result = self.chain({
                "query": query, 
                "context": context
            })
            return result["answer"]
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return "Извините, не удалось сгенерировать ответ на ваш вопрос."


class AnswerEvaluator:
    """Агент для оценки и улучшения качества ответа"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._setup_chain()

    def _setup_chain(self):
        template = """
        Оцени и улучши следующий ответ на вопрос пользователя.

        Вопрос: {query}
        Контекст: {context}
        Исходный ответ: {answer}

        Критерии оценки:
        1. Соответствие вопросу
        2. Полнота информации
        3. Использование контекста
        4. Точность и достоверность
        5. Ясность изложения

        Улучшенный ответ:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "context", "answer"]
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_key="improved_answer"
        )

    def evaluate_and_improve(self, query: str, context: str, answer: str) -> str:
        """Оценивает и улучшает ответ"""
        try:
            result = self.chain({
                "query": query, 
                "context": context, 
                "answer": answer
            })
            return result["improved_answer"]
        except Exception as e:
            logger.error(f"Ошибка при оценке и улучшении ответа: {e}")
            return answer


class RAGAgentOrchestrator:
    """Оркестратор для координации работы агентов"""

    def __init__(self, 
                 llm: BaseChatModel, 
                 chunk_searcher, 
                 sentence_transformer,
                 cohere_api_key: Optional[str] = None,
                 proxy_api_url: str = "https://api.proxyapi.ru/openai/v1/chat/completions"):
        self.llm = llm
        self.chunk_searcher = chunk_searcher
        self.sentence_transformer = sentence_transformer
        self.proxy_api_url = proxy_api_url

        # Инициализация агентов
        self.query_analyzer = QueryAnalyzer(llm)
        self.query_expander = QueryExpander(llm)
        self.document_reranker = DocumentReranker(api_key=cohere_api_key)
        self.answer_generator = AnswerGenerator(llm)
        self.answer_evaluator = AnswerEvaluator(llm)

    def process_query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Обрабатывает запрос пользователя с использованием мультиагентской системы"""

        logger.info(f"Обработка запроса: {query}")

        # Шаг 1: Анализ запроса
        query_analysis = self.query_analyzer.analyze(query)
        logger.debug(f"Анализ запроса: {query_analysis}")

        # Шаг 2: Расширение запроса
        expanded_query = self.query_expander.expand(query, query_analysis)
        logger.debug(f"Расширенный запрос: {expanded_query}")

        # Шаг 3: Поиск документов (используем существующий ChunkSearcher)
        # Сначала получаем больше документов для последующего переранжирования
        query_embedding = self.sentence_transformer.encode([query], convert_to_tensor=False)
        documents = self.chunk_searcher.search(np.array(query_embedding), k=k*3)  # получаем больше документов

        # Шаг 4: Переранжирование документов
        reranked_documents = self.document_reranker.rerank(query, documents)
        top_documents = reranked_documents[:k]  # берем только k лучших

        # Шаг 5: Подготовка контекста
        context = "\n\n".join([doc["text"] for doc in top_documents])

        # Шаг 6: Генерация ответа
        answer = self.answer_generator.generate(query, context)

        # Шаг 7: Оценка и улучшение ответа
        improved_answer = self.answer_evaluator.evaluate_and_improve(query, context, answer)

        # Формирование результата
        return {
            "question": query,
            "query_analysis": query_analysis,
            "expanded_query": expanded_query,
            "context": context,
            "context_used": [doc["text"][:200] + "..." for doc in top_documents],
            "initial_answer": answer,
            "final_answer": improved_answer
        }
