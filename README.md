# Customer Segmentation Pipeline with Apache Airflow

An end-to-end machine learning pipeline, orchestrated with **Apache Airflow 3** and running in **Docker**, that groups credit card customers into segments using **K-Means clustering**. The pipeline picks the number of segments automatically with the **elbow method**, evaluates the clusters, and assigns new customers to a segment.

This project is a modified version of [Airflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_1) from the MLOps course repository.

---

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Modifications from the Original Lab](#modifications-from-the-original-lab)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Running](#setup-and-running)
- [Outputs](#outputs)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

---

## Overview

| | |
|---|---|
| **Orchestration** | Apache Airflow 3 (Docker Compose) |
| **ML task** | Unsupervised clustering (customer segmentation) |
| **Model** | K-Means (`scikit-learn`) |
| **Choosing k** | Elbow method (`kneed`) |
| **Evaluation** | Silhouette score, cluster sizes, per-segment profiles |
| **Operators used** | `PythonOperator`, `BashOperator` |

---

## Key Concepts

**Apache Airflow** is a workflow orchestrator. You define a pipeline as a set of tasks and the order they run in, and Airflow schedules them, runs them, retries failures, and logs every run in a web UI.

**DAG (Directed Acyclic Graph).** A DAG is how Airflow represents a pipeline. Tasks are the nodes, and dependencies are one-directional arrows between them. The graph has no loops, so every run has a clear start and end.

**XCom (cross-communication).** XCom is how Airflow tasks pass small values to each other. A task's return value is stored automatically, and a later task reads it through `.output`. Because XCom lives in Airflow's metadata database, it is meant for small data such as file paths, IDs and metrics, not whole datasets.

**K-Means clustering** splits data into *k* groups by repeatedly assigning each point to its nearest cluster centre and moving each centre to the middle of its group.

**Elbow method.** SSE (the sum of squared distances from points to their cluster centre) always decreases as *k* grows. The "elbow" is the point where adding more clusters stops giving a large improvement, and it is a common choice for *k*.

**Silhouette score** measures how well separated the clusters are, on a scale from −1 to 1. Higher is better.

---

## Modifications from the Original Lab

| Area | Original lab | This version |
|---|---|---|
| Model used for predictions | Last model from the loop (k = 49); the elbow result was only printed | Final model trained with the elbow-selected k |
| Scaling of new data | `test.csv` predicted **unscaled**, although the model was trained on scaled data | Fitted scaler is saved and applied to new data |
| Data between tasks | Entire dataset pickled and base64-encoded into XCom | Data saved to files; only paths and small values go through XCom |
| Features | `BALANCE`, `PURCHASES`, `CREDIT_LIMIT` | Adds `PAYMENTS` |
| k search range | 1–49 | 1–15 |
| K-Means initialisation | `random` | `k-means++` |
| Evaluation | None | Silhouette score, cluster sizes and segment profiles |
| Predictions | First row of `test.csv` only | Every customer in a 5-row `test.csv`, saved to CSV |
| DAG structure | 4 tasks in a straight line | 7 tasks with parallel branches and a `BashOperator` |
| DAG settings | Defaults | Explicit `schedule=None`, 1 retry, tags, DAG docs in the UI |
| Input validation | None | Clear error if required columns are missing |

### Bug fix 1: the wrong model was saved

The original `build_save_model()` trained K-Means for k = 1 … 49 in a loop and saved whatever model was left in the loop variable, which was the **k = 49** model. The elbow method in the next task found the best k but only printed it, so the result was never used.

**Fix:** `compute_sse` builds the elbow curve, and `train_final_model` uses the elbow result to train the model that is saved and used for predictions.

### Bug fix 2: new data was not scaled

K-Means assigns a point to the nearest cluster centre. The centres were learned on features scaled to the 0–1 range, but the test customer was passed in raw units (for example, a credit limit of about 241,000). The distances, and therefore the predicted cluster, were meaningless.

**Fix:** `data_preprocessing` saves the fitted `MinMaxScaler`, and `predict_new_customers` uses it to scale new data before predicting.

### Design improvement: file paths instead of data in XCom

The original lab pushed the whole dataset through XCom as an encoded string. That works for a small CSV but does not scale. This version writes intermediate data to `dags/working_data/` and passes only the file path, which is the standard pattern in production pipelines (where the files would usually live in cloud storage).

---

## Pipeline Architecture

```
            load_data_task
                  │
        data_preprocessing_task
                  │
           compute_sse_task
                  │
        train_final_model_task
          ┌───────┴────────┐
evaluate_model_task   predict_new_customers_task
          └───────┬────────┘
        pipeline_summary_task
```

| Task | Operator | What it does | Output passed via XCom |
|---|---|---|---|
| `load_data_task` | Python | Reads `file.csv`, validates columns, drops missing values, saves `clean_data.csv` | Path to clean data |
| `data_preprocessing_task` | Python | Scales features with `MinMaxScaler`; saves `scaler.pkl` and the scaled data | Path to scaled data |
| `compute_sse_task` | Python | Fits K-Means for k = 1 … 15, records SSE, saves `elbow_curve.csv` | List of SSE values |
| `train_final_model_task` | Python | Finds the elbow with `kneed`, trains the final model, saves `kmeans_model.pkl` | `{optimal_k, model_path}` |
| `evaluate_model_task` | Python | Computes the silhouette score, cluster sizes and segment profiles | Metrics dictionary |
| `predict_new_customers_task` | Python | Scales `test.csv` with the saved scaler and assigns each customer a cluster | List of cluster IDs |
| `pipeline_summary_task` | Bash | Prints the metrics, profiles and predictions into the task log | — |

`evaluate_model_task` and `predict_new_customers_task` both depend only on the trained model, not on each other, so Airflow runs them **in parallel** (fan-out). `pipeline_summary_task` waits for both to finish (fan-in).

---

## Dataset

**`dags/data/file.csv`** contains about 8,950 credit card customers with 18 columns describing their account behaviour. The pipeline uses four of them:

| Feature | Description |
|---|---|
| `BALANCE` | Amount owed on the account |
| `PURCHASES` | Total amount spent on purchases |
| `CREDIT_LIMIT` | Credit limit of the card |
| `PAYMENTS` | Total amount paid back |

**`dags/data/test.csv`** contains five new customers (`NEW_001` to `NEW_005`) with the same four features. The pipeline assigns each of them to a segment.

---

## Project Structure

```
.
├── dags/
│   ├── airflow.py              # DAG definition
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py              # Task functions
│   ├── data/
│   │   ├── file.csv            # Training data
│   │   └── test.csv            # New customers to segment
│   ├── model/                  # Created at runtime (git-ignored)
│   └── working_data/           # Created at runtime (git-ignored)
├── config/                     # Generated airflow.cfg
├── docker-compose.yaml         # Airflow 3 services
├── setup.sh                    # Resets the environment and writes .env
└── README.md
```

---

## Setup and Running

### Prerequisites

- **Docker Desktop**, installed and running
- At least **4 GB of memory** allocated to Docker (8 GB recommended), set under *Docker Desktop → Settings → Resources*

### 1. Clone the repository

```bash
git clone https://github.com/Kirtan10/MLOps-Airflow-Lab1.git
cd MLOps-Airflow-Lab1
```

### 2. Get the Airflow Docker Compose file

```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
```

### 3. Configure `docker-compose.yaml`

In the `environment` section under `x-airflow-common`, set:

```yaml
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:-scikit-learn kneed}
```

The first line hides Airflow's example DAGs. The second installs the libraries this pipeline needs when the containers start.

### 4. Prepare the environment

```bash
bash setup.sh
```

This script removes any old containers, volumes and logs, recreates the `logs/`, `plugins/` and `config/` folders, and writes your user ID to `.env`, so the files Airflow creates are owned by you.

### 5. Initialise and start Airflow

```bash
docker compose up airflow-init
docker compose up
```

### 6. Open the Airflow UI

Go to **http://localhost:8080** and log in. The default username and password are both `airflow`.

### 7. Trigger the DAG

1. Find **`Airflow_Lab1_Customer_Segmentation`** in the DAG list.
2. Click **Trigger**.
3. Open the run to watch the tasks progress in the Graph view.
4. Open the **`pipeline_summary_task`** log to see the final results.

You can also run the whole DAG from the command line:

```bash
docker compose run --rm airflow-cli airflow dags test Airflow_Lab1_Customer_Segmentation
```

### 8. Stop Airflow

```bash
docker compose down
```

---

## Outputs

After a successful run, these files are created:

| File | Location | Contents |
|---|---|---|
| `scaler.pkl` | `dags/model/` | Fitted `MinMaxScaler` |
| `kmeans_model.pkl` | `dags/model/` | Final K-Means model |
| `clean_data.csv` | `dags/working_data/` | Data after dropping missing values |
| `scaled_data.pkl` | `dags/working_data/` | Scaled feature matrix |
| `elbow_curve.csv` | `dags/working_data/` | SSE for each k |
| `metrics.json` | `dags/working_data/` | Optimal k, silhouette score, inertia, cluster sizes |
| `cluster_profiles.csv` | `dags/working_data/` | Mean feature values and size of each cluster |
| `predictions.csv` | `dags/working_data/` | Assigned cluster for each customer in `test.csv` |

---

## Results

Results from a sample run (values may differ slightly between library versions):

- **Optimal number of clusters:** 4
- **Silhouette score:** 0.48

| Cluster | Mean balance | Mean purchases | Mean credit limit | Mean payments | Customers |
|---|---|---|---|---|---|
| 0 | 4,099 | 1,069 | 6,698 | 2,384 | 1,255 |
| 1 | 711 | 1,743 | 8,462 | 2,507 | 1,538 |
| 2 | 7,812 | 3,616 | 12,855 | 6,558 | 430 |
| 3 | 769 | 594 | 2,318 | 1,021 | 5,726 |

**Interpreting the segments:**

- **Cluster 3: low-activity customers.** This is the largest group, with low limits, balances and spending.
- **Cluster 1: active spenders.** They spend regularly but keep low balances.
- **Cluster 0: balance carriers.** They hold high balances relative to their purchases.
- **Cluster 2: high-value customers.** This small group has high limits, balances and spending.

---

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'kneed'` (or `sklearn`) | The libraries were not installed in the containers. Check `_PIP_ADDITIONAL_REQUIREMENTS` in `docker-compose.yaml`, then restart with `docker compose down` and `docker compose up`. |
| The DAG does not appear in the UI | It can take a minute for new DAGs to be picked up. If it still does not appear, look for an import error banner at the top of the DAG list. |
| Containers keep restarting or exit | Docker probably does not have enough memory. Increase it to at least 4 GB in Docker Desktop settings. |
| Port 8080 is already in use | Stop the other service using the port, or change the port mapping in `docker-compose.yaml`. |
| Permission errors on `logs/` or `dags/` | Run `bash setup.sh` again so `.env` contains your user ID. |

---

## Credits

Based on Airflow Lab 1 from [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps).