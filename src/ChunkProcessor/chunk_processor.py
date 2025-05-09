import logging
import numpy as np
from typing import List, Dict, Any

from db_chunk_utils.raw_data_fetcher import RawDataFetcher
from db_chunk_utils.chunks_db import save_chunks_batch, article_exists
from db_chunk_utils.base import get_chunks_db_connection

from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkProducer:
    def __init__(self, words_per_chunk: int = 500, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.words_per_chunk = words_per_chunk
        self.fetcher = RawDataFetcher()
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Генерирует эмбеддинг для текста с помощью модели.
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Ошибка при генерации эмбеддинга для текста: {e}")
            return np.zeros(384)

    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Делит текст на части по словам.
        """
        words = text.split()
        return [
            " ".join(words[i:i + self.words_per_chunk])
            for i in range(0, len(words), self.words_per_chunk)
        ]

    def _prepare_metadata(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Формирует метаданные для каждого чанка статьи.
        """
        return [{
            "header": article["header"],
            "url": article["url"],
            "tags": article.get("tags", []),
            "time": article.get("time"),
            "created_at": article.get("created_at")
        } for _ in range(len(article["_chunks"]))]

    def process_articles(self, articles: List[Dict[str, Any]]):
        """
        Основная логика обработки статей: деление на чанки, создание эмбеддингов, сохранение.
        """
        conn = None
        try:
            conn = get_chunks_db_connection()

            for article in articles:
                logger.info(f"Обрабатываем статью ID={article['id']}, Заголовок='{article['header']}'")

                # Разбиваем текст на чанки
                chunks = self.split_text_into_chunks(article["text"])

                if not chunks:
                    logger.warning(f"Статья {article['id']} не содержит текста для разбиения.")
                    continue

                if article_exists(conn, article['id']):
                    logger.info(f"Статья ID={article['id']} уже существует. Пропускаем.")
                    continue

                article["_chunks"] = chunks

                embeddings = [self.generate_embedding(chunk) for chunk in chunks]

                metadatas = self._prepare_metadata(article)

                save_chunks_batch(conn, article, chunks, embeddings, metadatas)

        except Exception as e:
            logger.error(f"Ошибка при обработке статей: {e}")
        finally:
            if conn and not conn.closed:
                conn.close()
                logger.info("Соединение с chunks_db закрыто")

    def run(self):
        """
        Точка входа в работу ChunkProducer.
        """
        articles = self.fetcher.fetch_latest_articles()

        if articles:
            logger.info(f"Получено {len(articles)} статей для обработки.")
            self.process_articles(articles)
        else:
            logger.info("Новых статей нет.")


if __name__ == "__main__":
    producer = ChunkProducer(words_per_chunk=500)
    producer.run()