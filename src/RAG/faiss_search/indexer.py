import psycopg2
import os
import logging
import numpy as np
import faiss
import dotenv
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class FaissIndexer():
    def __init__(self, index_path, ids_path):
        self.chunks_db_config = {
            'host': os.getenv("CHUNKS_DB_HOST", "localhost"),
            'port': int(os.getenv("CHUNKS_DB_PORT", 5432)),
            'database': os.getenv("CHUNKS_DB_NAME", "chunks_db"),
            'user': os.getenv("CHUNKS_DB_USER", "postgres"),
            'password': os.getenv("CHUNKS_DB_PASSWORD", "password")
        }

        self.index_path = index_path
        self.ids_path = ids_path



    def _get_chunks_db_connection(self):
        """Устанавливает подключение к бд"""

        try:
            conn = psycopg2.connect(**self.chunks_db_config)
            logger.info("Подключились к chunks_db")
            return conn
        except Exception as e:
            logger.error(f"Ошибка подключения к chunks_db: {e}")
            raise

    def _get_embeddings_from_db(self):
        """Получает бинарные эмбеддинги из БД и возвращает их в виде массива numpy"""

        ids = []
        embeddings = []
        conn = None

        try:
            conn = self._get_chunks_db_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT id, embedding FROM chunks")
                rows = cur.fetchall()

                for row in rows:
                    chunk_id = row[0]
                    embedding_bytes = row[1]

                    try:
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        ids.append(chunk_id)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Ошибка при обработке эмбеддинга ID {chunk_id}: {e}")
                        continue

                embeddings_np = np.array(embeddings, dtype=np.float32)

                return ids, embeddings_np

        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов из БД: {e}")
            return [], np.array([])

        finally:
            if conn is not None:
                conn.close()

    def build_and_save_index(self):
        """
        Строит FAISS-индекс из эмбеддингов из БД и сохраняет его на диск.
        Также сохраняет соответствие индексов векторов оригинальным ID чанков.
        Возвращает путь к файлу индекса или None при ошибке.
        """

        logger.info("Начало построения FAISS-индекса")

        ids, embeddings = self._get_embeddings_from_db()

        if embeddings is None or embeddings.shape[0] == 0:
            logger.warning("Нет данных для построения индекса.")
            return None

        dimension = embeddings.shape[1]
        logger.info(f"Построение индекса из {len(ids)} записей, размерность: {dimension}")

        index = faiss.IndexFlatL2(dimension)
        logger.debug("Создан базовый индекс IndexFlatL2")

        try:
            index.add(embeddings)
            logger.debug("Эмбеддинги успешно добавлены в индекс")
        except Exception as e:
            logger.error(f"Ошибка при добавлении эмбеддингов в индекс: {e}", exc_info=True)
            return None

        try:
            faiss.write_index(index, self.index_path)
            logger.info(f"Индекс успешно сохранён в файл: {self.index_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении индекса: {e}", exc_info=True)
            return None

        try:
            with open(self.ids_path, 'wb') as f:
                pickle.dump(ids, f)
            logger.info(f"ID чанков успешно сохранены в файл: {self.ids_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении ID чанков: {e}", exc_info=True)
            return None

        return self.index_path

