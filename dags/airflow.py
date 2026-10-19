"""
### Customer Segmentation Pipeline (Airflow Lab 1, modified)

Clusters credit card customers with K-Means, choosing the number of clusters
with the elbow method, then evaluates the clusters and assigns new customers
to a segment.

Task flow:
load_data -> data_preprocessing -> compute_sse -> train_final_model
-> [evaluate_model, predict_new_customers] -> pipeline_summary
"""
from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

from src.lab import (
    OUTPUT_DIR,
    compute_sse,
    data_preprocessing,
    evaluate_model,
    load_data,
    predict_new_customers,
    train_final_model,
)

default_args = {
    "owner": "Kirtan Patel",
    "retries": 1,                          # MODIFIED: retry a failed task once
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="Airflow_Lab1_Customer_Segmentation",
    default_args=default_args,
    description="K-Means customer segmentation with elbow method, evaluation and prediction",
    start_date=datetime(2026, 1, 1),
    schedule=None,                         # manual trigger only
    catchup=False,
    tags=["mlops", "clustering", "lab1"],
    doc_md=__doc__,                        # shows the docstring above in the Airflow UI
) as dag:

    # Each task receives the previous task's return value through XCom (.output).
    # Only small values are passed this way: file paths, the SSE list and a small dict.

    load_data_task = PythonOperator(
        task_id="load_data_task",
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id="data_preprocessing_task",
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    compute_sse_task = PythonOperator(
        task_id="compute_sse_task",
        python_callable=compute_sse,
        op_args=[data_preprocessing_task.output],
    )

    train_final_model_task = PythonOperator(
        task_id="train_final_model_task",
        python_callable=train_final_model,
        op_args=[data_preprocessing_task.output, compute_sse_task.output],
    )

    # NEW: evaluate the clusters (silhouette score + segment profiles)
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model_task",
        python_callable=evaluate_model,
        op_args=[
            load_data_task.output,
            data_preprocessing_task.output,
            train_final_model_task.output,
        ],
    )

    predict_task = PythonOperator(
        task_id="predict_new_customers_task",
        python_callable=predict_new_customers,
        op_args=[train_final_model_task.output],
    )

    # NEW: a BashOperator that prints the final outputs into the task log
    pipeline_summary_task = BashOperator(
        task_id="pipeline_summary_task",
        bash_command=(
            f"echo '===== Metrics ====='; cat {OUTPUT_DIR}/metrics.json; echo; "
            f"echo '===== Cluster profiles ====='; cat {OUTPUT_DIR}/cluster_profiles.csv; "
            f"echo '===== Predictions ====='; cat {OUTPUT_DIR}/predictions.csv"
        ),
    )

    # MODIFIED: evaluation and prediction are independent, so they run in parallel
    # after training (fan-out), and the summary waits for both (fan-in).
    (
        load_data_task
        >> data_preprocessing_task
        >> compute_sse_task
        >> train_final_model_task
        >> [evaluate_model_task, predict_task]
        >> pipeline_summary_task
    )

if __name__ == "__main__":
    dag.test()