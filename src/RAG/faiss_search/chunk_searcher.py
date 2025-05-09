# src/retriever/chunk_searcher.py

import os
import logging
import numpy as np
import psycopg2

logger = logging.getLogger(__name__)


class ChunkSearcher:
    def __init__(self,
                 index_path: str = None,
                 idmap_path: str = None,
                 db_config: dict = None):
        """
        :param index_path: путь к .bin файлу FAISS-индекса
        :param idmap_path: путь к .idmap файлу
        :param db_config: словарь с настройками подключения к БД
        """

        self.index_path = index_path or "data/faiss_index.bin"
        self.idmap_path = idmap_path or "data/ids.idmap"

        self.db_config = db_config or {
            'host': os.getenv("CHUNKS_DB_HOST", "localhost"),
            'port': int(os.getenv("CHUNKS_DB_PORT", 5432)),
            'database': os.getenv("CHUNKS_DB_NAME", "chunks_db"),
            'user': os.getenv("CHUNKS_DB_USER", "postgres"),
            'password': os.getenv("CHUNKS_DB_PASSWORD", "password")
        }

        # Создаем FaissRetriever с указанными путями
        from .faiss_retriever import FaissRetriever
        self.retriever = FaissRetriever(
            index_path=self.index_path,
            idmap_path=self.idmap_path,
            build_if_missing=True
        )

    def _connect(self):
        """Устанавливает подключение к PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            logger.info("Подключились к базе данных чанков")
            return conn
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise

    def get_chunk_details(self, chunk_id: int) -> dict:
        """Получает текст чанка и его метаданные из базы данных"""
        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT chunk_text, created_at, article_id, metadata 
                    FROM chunks WHERE id = %s
                """, (chunk_id,))
                result = cur.fetchone()
                if result:
                    text, created_at, article_id, metadata = result
                    return {
                        "text": text,
                        "created_at": created_at,
                        "article_id": article_id,
                        "metadata": metadata
                    }
                else:
                    logger.warning(f"Чанк с ID={chunk_id} не найден в БД")
                    return {}
        except Exception as e:
            logger.error(f"Ошибка при получении деталей чанка: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """
        Выполняет поиск ближайших чанков по эмбеддингу запроса

        :param query_embedding: векторный запрос (np.ndarray размерности D)
        :param k: количество результатов
        :return: список словарей с текстом чанка и метаданными
        """
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("query_embedding должен быть типа np.ndarray")

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        results = self.retriever.retrieve(query_embedding, k=k)

        matched_chunks = []
        for chunk_id, score in results:
            chunk_details = self.get_chunk_details(chunk_id)
            if chunk_details:
                chunk_details["score"] = float(score)
                matched_chunks.append(chunk_details)

        return matched_chunks