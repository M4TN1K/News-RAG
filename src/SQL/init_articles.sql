CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    header TEXT,
    time DATE,
    text TEXT,
    tags TEXT[],
    text_len_words INTEGER,
    text_len_sym INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
ALTER DATABASE news_parser SET TIMEZONE TO 'UTC';