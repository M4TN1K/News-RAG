from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime

with (DAG(
    dag_id="news_preprocessing_dag",
    start_date=datetime(2025, 5, 5),
    schedule_interval=None,
    catchup=False,
) as dag):

    run_parser = DockerOperator(
        task_id='run_parser',
        image='parser:latest',
        api_version='auto',
        auto_remove='success',
        command='python parser.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='rbc_network',
    )

    run_chunk_processor = DockerOperator(
        task_id='run_chunk_processor',
        image='chunk_processor:latest',
        api_version='auto',
        auto_remove='success',
        command='python chunk_processor.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='rbc_network',
    )

    run_parser >> run_chunk_processor