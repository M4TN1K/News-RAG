# db_utils/raw_data_fetcher.py

import logging
from typing import List, Dict
from datetime import datetime, timedelta, date
from psycopg2 import sql



from db_chunk_utils.base import get_main_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RawDataFetcher:
    def __init__(self):
        self.conn = None
        self.time_thershold = 3

    def _connect(self):
        """Устанавливает соединение с БД через служебную функцию"""
        if self.conn is None or self.conn.closed != 0:
            try:
                self.conn = get_main_db_connection()
                logger.info("Подключились к articles")
            except Exception as e:
                logger.error(f"Не удалось установить соединение с БД: {e}")
                raise

    def _get_cutoff_time(self) -> datetime:
        """
        Возвращает время, на которое нужно фильтровать (последние 3 часа)
        """
        return datetime.utcnow() - timedelta(hours=self.time_thershold)

    def _fetch_articles(self, cutoff_time: datetime) -> List[Dict]:
        """
        Загружает последние добавленные статьи
        """
        cur = self.conn.cursor()
        try:
            query = sql.SQL("""
                SELECT id, url, header, text, tags, time, created_at
                FROM articles
                WHERE created_at > %s
                ORDER BY created_at DESC;
            """)
            cur.execute(query, (cutoff_time,))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

            results = []
            for row in rows:
                article = dict(zip(columns, row))
                if isinstance(article.get('created_at'), datetime):
                    article['created_at'] = article['created_at'].isoformat()
                if isinstance(article.get('time'), date):
                    article['time'] = article['time'].isoformat()
                results.append(article)
            return results
        except Exception as e:
            logger.error(f"Ошибка при загрузке статей: {e}")
            return []
        finally:
            cur.close()

    def fetch_latest_articles(self) -> List[Dict]:
        """
        Основной метод: возвращает статьи, добавленные за последние 3 часа
        """
        try:
            self._connect()
            cutoff_time = self._get_cutoff_time()
            logger.info(f"Загрузка статей, добавленных после {cutoff_time.isoformat()}")

            return self._fetch_articles(cutoff_time)
        except Exception as e:
            logger.error(f"Критическая ошибка при работе с БД: {e}")
            return []
        finally:
            if self.conn and not self.conn.closed:
                self.conn.close()
                logger.info("Соединение с articles закрыто")

