
FROM python:3.9-slim-buster

RUN apt-get update && \
    apt-get install -y \
        wget \
        gnupg \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        fonts-liberation \
        libnss3 \
        libasound2 \
        firefox-esr \
        unzip && \
    rm -rf /var/lib/apt/lists/*

ARG GECKO_DRIVER_VERSION=0.34.0
RUN echo "Скачивание GeckoDriver..." && \
    wget -O /tmp/geckodriver.tar.gz https://github.com/mozilla/geckodriver/releases/download/v${GECKO_DRIVER_VERSION}/geckodriver-v${GECKO_DRIVER_VERSION}-linux64.tar.gz && \
    tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin/ && \
    chmod +x /usr/local/bin/geckodriver && \
    echo "GeckoDriver установлен"


WORKDIR /app

COPY . /app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]