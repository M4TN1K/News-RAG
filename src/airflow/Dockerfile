
FROM apache/airflow:2.10.5


USER airflow

COPY requirements.txt /tmp/requirements.txt


RUN pip install -r /tmp/requirements.txt
USER airflow