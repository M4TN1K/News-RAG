CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    article_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    metadata JSONB DEFAULT '{}'::JSONB

);