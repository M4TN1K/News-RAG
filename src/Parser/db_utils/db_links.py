import psycopg2

import logging

logger = logging.getLogger(__name__)

class DatabaseLinksManager:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cur = None

    def _connect_db(self):
        """Подключение к PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor()
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise

    def _close_db(self):
        """Закрытие соединения с БД"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def load_links_from_db(self):
        """Загрузка всех ссылок из БД"""
        try:
            self._connect_db()
            self.cur.execute("SELECT url FROM links ORDER BY created_at DESC;")
            rows = self.cur.fetchall()
            loaded_links = [row[0] for row in rows]
            logger.info(f"Загружено {len(loaded_links)} ссылок из БД")
            return loaded_links
        except Exception as e:
            logger.error(f"Ошибка загрузки из БД: {e}")
            return []
        finally:
            self._close_db()

    def save_links_to_db(self, loaded_links, original_loaded_links):
        """Удаление старых ссылок и сохранение новых"""
        try:
            self._connect_db()
            # Удаляем старые данные
            self.cur.execute("DELETE FROM links;")

            new_links = set(loaded_links) - set(original_loaded_links)

            if len(new_links) != 0:
                logger.info(f"Найдено {len(new_links)} новых ссылок. Обновляем таблицу.")
                self.cur.execute("TRUNCATE TABLE links;")

            for link in new_links:
                self.cur.execute(
                    """
                    INSERT INTO links (url)
                    VALUES (%s)
                    ON CONFLICT (url) DO NOTHING;
                    """,
                    (link,)
                )
            self.conn.commit()
            logger.info(f"Сохранено {len(new_links)} новых ссылок в БД")
        except Exception as e:
            logger.error(f"Ошибка сохранения в БД: {e}")
        finally:
            self._close_db()