from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_hello():
    print("Hello, World!")

# Define the DAG
with DAG(
    "hello_world_dag",
    schedule_interval=None,  # None means it won't be scheduled automatically
    start_date=datetime(2025, 2, 17),  # Start date for the DAG
    catchup=False,
) as dag:
    hello_task = PythonOperator(
        task_id="hello_task",
        python_callable=print_hello,
    )

    hello_task  # Set the task
