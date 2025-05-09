import psycopg2
from psycopg2 import sql, extras
import logging

logger = logging.getLogger(__name__)

class DatabaseNewsManager:
    def __init__(self, db_config):
        self.db_config = db_config

    def save_news_batch_to_db(self, news_list: list):
        """
        Сохраняет список статей в таблицу articles.
        Если запись с таким url уже существует — обновляет её.

        :param news_list: Список словарей с данными о статьях
        """
        if not news_list:
            logger.info("Нет данных для сохранения в БД")
            return

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            insert_query = sql.SQL("""
                INSERT INTO articles (
                    url,
                    header,
                    time,
                    text,
                    tags,
                    text_len_words,
                    text_len_sym
                ) VALUES (
                    %(news_link)s,
                    %(header)s,
                    %(time)s,
                    %(text)s,
                    %(tags)s,
                    %(text_len_words)s,
                    %(text_len_sym)s
                )
                ON CONFLICT (url) DO UPDATE SET
                    header = EXCLUDED.header,
                    time = EXCLUDED.time,
                    text = EXCLUDED.text,
                    tags = EXCLUDED.tags,
                    text_len_words = EXCLUDED.text_len_words,
                    text_len_sym = EXCLUDED.text_len_sym;
            """)

            # Выполняем пакетное сохранение
            extras.execute_batch(cur, insert_query, news_list, page_size=100)
            conn.commit()
            logger.info(f"Сохранено {len(news_list)} статей в БД")

        except Exception as e:
            logger.error(f"Ошибка при пакетном сохранении в БД: {e}")
            conn.rollback()
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()