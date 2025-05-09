import signal
import os
import time
import datetime
import random
from bs4 import BeautifulSoup
import logging
from typing import Tuple, List, Dict, Optional, Union

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from shutil import which

import re
import requests
from db_utils.db_links import DatabaseLinksManager
from db_utils.db_articles import DatabaseNewsManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../scraper.log"),  # Логи в файл
        logging.StreamHandler()  # Логи в консоль
    ]
)
logger = logging.getLogger(__name__)


class RbcNewsParser:
    def __init__(self, url: str):
        self.url = url
        self.running = True
        self.loaded_links = []  # Буфер для ссылок
        self.original_loaded_links = []  # Оригинальные ссылки из БД
        self.news = []  # Список кортежей (ссылка, текст новости)
        self.scroll_count = 0
        self.max_scrolls = 1  # Максимум прокруток страницы
        self.driver = None
        self.batch_size = 5  # Размер батча для сохранения в БД
        self.flag_of_the_old_articles_existing = False  # Флаг того ,что есть старые ссылки в выдаче парсера



        self.db_config = {
            "host": os.getenv("DB_HOST", "rbc_postgres"),
            "port": os.getenv("DB_PORT", 5432),
            "database": os.getenv("DB_NAME", "news_parser"),
            "user": os.getenv("DB_USER", "parser_user"),
            "password": os.getenv("DB_PASSWORD", "parser_pass")
        }

        self.db_link_manager = DatabaseLinksManager(self.db_config)
        self.db_news_manager = DatabaseNewsManager(self.db_config)


        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_links_from_db(self):
        """Загрузка первых 100 ссылок из БД"""
        self.loaded_links = self.db_link_manager.load_links_from_db()
        self.original_loaded_links = self.loaded_links.copy()

    def _save_links_to_db(self):
        """Сохранение новых ссылок в БД"""
        self.db_link_manager.save_links_to_db(self.loaded_links, self.original_loaded_links)



    def _signal_handler(self, signum, frame):
        """Обработчик системных сигналов остановки"""
        logger.info("Получен сигнал остановки. Завершаем работу...")
        self.running = False
        self._shutdown()

    def _shutdown(self):
        """Очистка ресурсов перед завершением работы"""
        if self.driver:
            self.driver.quit()
            self.driver = None
        logger.info("Парсер корректно выключен")

    def _kill_process(self) -> bool:
        """Проверяет, нужно ли остановить процесс"""
        return not self.running or self.driver is None

    def _setup_driver(self) -> bool:
        """Инициализация и настройка WebDriver (Firefox)."""
        try:
            if not which("geckodriver"):
                logger.error("GeckoDriver не найден в PATH")
                return False


            options = FirefoxOptions()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")


            driver_path = os.getenv('GECKODRIVER_PATH', '/usr/local/bin/geckodriver')
            service = FirefoxService(executable_path=driver_path)
            #service = FirefoxService(executable_path='/usr/local/bin/geckodriver')

            self.driver = webdriver.Firefox(service=service, options=options)
            self.driver.get(self.url)
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации драйвера Firefox: {e}")
            return False

    def _webpage_scrapper(self, url: str) -> Dict[str, Optional[Union[List, str, int]]]:
        """
        Парсит веб-страницу и извлекает очищенный текст из тегов <p>, <li> и <h1>.

        Args:
            url (str): URL веб-страницы для парсинга

        Returns:
            str: Очищенный текст, объединенный в одну строку

        Raises:
            requests.exceptions.RequestException: При проблемах с HTTP-запросом
            Exception: При других ошибках парсинга
        """
        try:
            if self._kill_process():
                return ""

            logging.info(f"Начинаем парсинг URL: {url}")

            # Отправляем HTTP-запрос с проверкой ответа
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            logging.info(f"Получен ответ сервера: {response.status_code}")

            text_dict = {}

            soup = BeautifulSoup(response.text, 'html.parser')

            header = soup.find_all('h1', recursive=True)[0].get_text()
            cleaned_header = re.sub(r'[\n\xa0]', ' ', header)
            cleaned_header = re.sub(r'\s+', ' ', cleaned_header).strip()

            current_time = datetime.datetime.now().strftime('%Y-%m-%d')

            tags = soup.find_all('a', class_='article__tags__item', recursive=True)
            tags = [tag.get_text() for tag in tags]

            raw_text = soup.find_all(['p', 'li'], recursive=True)
            raw_text = [string.get_text() for string in raw_text]

            preprocess_text = []
            for string in raw_text:
                if len(string.split()) > 30:
                    # Очищаем от лишних пробелов и непечатных символов
                    cleaned = re.sub(r'[\n\xa0]', ' ', string)
                    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

                    if cleaned:
                        preprocess_text.append(cleaned)

            full_text = ' '.join(preprocess_text)
            text_dict['news_link'] = url
            text_dict['header'] = cleaned_header
            text_dict['time'] = current_time
            text_dict['text'] = full_text
            text_dict['tags'] = tags
            text_dict['text_len_words'] = len(full_text.split())
            text_dict['text_len_sym'] = len(full_text)

            return text_dict

        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка HTTP-запроса: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Произошла ошибка парсинга: {str(e)}", exc_info=True)
            raise
        finally:
            logging.info("Завершение работы парсера")

    def _scroll_page(self):
        """Интеллектуальная прокрутка страницы."""
        try:
            if self._kill_process():
                return

            for _ in range(random.randint(1, 3)):
                if self._kill_process():
                    break


                self.driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight);"
                )

                time.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.warning(f"Ошибка прокрутки: {e}")

    def _parse_page(self) -> bool:
        """Парсинг текущего состояния страницы."""
        try:
            if self._kill_process():
                return False

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            new_articles = soup.find_all('div', class_='article-card')

            current_links = [
                a['data-metronome-href'] for a in new_articles
                if a.has_attr('data-metronome-href')
            ]

            new_links = []
            for link in current_links:
                if link and link not in self.loaded_links:
                    new_links.append(link)
                else:
                    self.flag_of_the_old_articles_existing = True

            batch = []

            if new_links:
                self.loaded_links.extend(new_links)

                for link in new_links:
                    try:
                        news_data = self._webpage_scrapper(link)

                        self.news.append(news_data)
                        batch.append(news_data)

                        if len(batch) >= self.batch_size:
                            self.db_news_manager.save_news_batch_to_db(batch)
                            batch.clear()

                    except Exception as e:
                        logger.error(f"Ошибка при парсинге {link}: {e}")

                    time.sleep(random.uniform(1, 3))

                if batch:
                    self.db_news_manager.save_news_batch_to_db(batch)
                    batch.clear()

            else:
                logger.info("Новых элементов не обнаружено")
                return False

            return True

        except Exception as e:
            logger.error(f"Ошибка парсинга: {e}")
            return False

    def start(self) -> List[Tuple[str, str]]:
        """Основной метод запуска парсера."""
        self._load_links_from_db()

        # Инициализируем вебдрайвер
        if not self._setup_driver():
            return []

        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )

            while self.running and self.scroll_count < self.max_scrolls:
                self._scroll_page()

                if self._kill_process():
                    break

                if not self._parse_page():
                    break

                self.scroll_count += 1

                logger.info(
                    f"Прокрутка {self.scroll_count}/{self.max_scrolls} завершена. "
                    f"Собрано ссылок: {len(self.loaded_links)}, "
                    f"Обработано новостей: {len(self.news)}"
                )

        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")

        finally:
            self._save_links_to_db()

            if self.driver:
                self.driver.quit()
                self.driver = None

            logger.info("Драйвер успешно закрыт")

        return self.news


if __name__ == "__main__":
    parser = RbcNewsParser('https://www.rbc.ru/industries/newsfeed')

    try:
        result = parser.start()
        logger.info("Работа завершена!")
    except KeyboardInterrupt:
        logger.info("Работа прервана пользователем")