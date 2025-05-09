import os
import logging
import psycopg2
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def get_main_db_connection():
    """Подключение к основной БД (сырые данные)"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "rbc_postgres"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "news_parser"),
            user=os.getenv("DB_USER", "parser_user"),
            password=os.getenv("DB_PASSWORD", "parser_pass")
        )
        logger.info("Подключились к main_db")
        return conn
    except Exception as e:
        logger.error(f"Ошибка подключения к main_db: {e}")
        raise

def get_chunks_db_connection():
    """Подключение к БД чанков"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("CHUNKS_DB_HOST", "chunks-db"),
            port=int(os.getenv("CHUNKS_DB_PORT", 5432)),
            database=os.getenv("CHUNKS_DB_NAME", "chunk_db"),
            user=os.getenv("CHUNKS_DB_USER", "chunk_user"),
            password=os.getenv("CHUNKS_DB_PASSWORD", "chunk_pass")
        )
        logger.info("Подключились к chunks_db")
        return conn
    except Exception as e:
        logger.error(f"Ошибка подключения к chunks_db: {e}")
        raise