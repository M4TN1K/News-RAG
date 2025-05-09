import psycopg2.extras
import numpy as np
import logging
from typing import Dict, List, Union
from psycopg2 import sql

logger = logging.getLogger(__name__)

def save_chunk(conn, article_id: int, chunk_index: int, chunk_text: str, embedding: np.ndarray, metadata: Dict[str, Union[List, str, float]]):
    """Сохраняет один чанк в таблицу chunks"""
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql.SQL("""
            INSERT INTO chunks (
                article_id,
                chunk_index,
                chunk_text,
                embedding,
                metadata
            ) VALUES (%s, %s, %s, %s, %s)
        """), (
            article_id,
            chunk_index,
            chunk_text,
            psycopg2.Binary(embedding.tobytes()),
            psycopg2.extras.Json(metadata)
        ))
        conn.commit()
        logger.debug(f"Чанк {chunk_index} статьи {article_id} сохранён")
    except Exception as e:
        logger.error(f"Ошибка при сохранении чанка: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()

def save_chunks_batch(conn, article_data: dict, chunks: list, embeddings: list, metadatas: list):
    """Сохраняет батч чанков одной статьи"""
    for idx, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadatas)):
        save_chunk(conn, article_data['id'], idx, chunk, embedding, metadata)

def article_exists(conn, article_id: int) -> bool:
    """
    Проверяет, существует ли статья с данным ID в таблице chunks.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chunks WHERE article_id = %s LIMIT 1", (article_id,))
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Ошибка при проверке существования статьи: {e}")
        return False