# src/retriever/faiss_retriever.py

import numpy as np
import faiss
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


from .indexer import FaissIndexer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class FaissRetriever:
    def __init__(self, index_path: str = None, idmap_path: str = None, build_if_missing: bool = True):
        """
        :param index_path: путь к .bin файлу FAISS-индекса
        :param idmap_path: путь к .idmap файлу с ID статей
        :param build_if_missing: строить ли индекс, если файлов нет
        """
        self.index_path = index_path or "faiss_index.bin"
        self.ids_path = idmap_path or "ids.idmap"
        self.build_if_missing = build_if_missing

        self.index = None
        self.id_map = None

        self._load_or_build_index()

    def _load_or_build_index(self):
        """Загружает индекс или строит его, если файлов нет"""
        index_path = Path(self.index_path)
        idmap_path = Path(self.ids_path)

        if not index_path.exists() or not idmap_path.exists():
            if self.build_if_missing:
                logger.warning("Файл(ы) индекса не найдены. Строим новый индекс...")
                self._build_index()
            else:
                raise FileNotFoundError("Файл(ы) индекса не найдены, а автоматическое создание отключено.")
        else:
            self._load_index()

    def _load_index(self):
        """Загружает существующий FAISS-индекс и соответствие ID"""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.ids_path, 'rb') as f:
                self.id_map = pickle.load(f)
            logger.info("FAISS-индекс и ID успешно загружены")
        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса или ID: {e}")
            raise

    def _build_index(self):
        """Строит новый индекс с помощью FaissIndexer"""
        try:
            indexer = FaissIndexer(index_path=self.index_path, ids_path=self.ids_path)
            indexer.build_and_save_index()
            self._load_index()
        except Exception as e:
            logger.error(f"Не удалось построить индекс: {e}")
            raise

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> list:
        """
        Выполняет поиск ближайших статей по эмбеддингу

        :param query_embedding: векторный запрос (np.ndarray размерности D)
        :param k: количество результатов
        :return: список кортежей (article_id, score)
        """
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("query_embedding должен быть типа np.ndarray")

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            article_id = self.id_map[idx]
            results.append((article_id, float(score)))

        return results

