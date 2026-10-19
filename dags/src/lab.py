"""
Customer segmentation pipeline functions for Airflow Lab 1.

Each function is one Airflow task. Large data (DataFrames, arrays, models) is
written to disk, and only small values (file paths, lists, small dicts) are
returned. Airflow stores return values in XCom, which is designed for small
pieces of data rather than whole datasets.
"""
import json
import os
import pickle

import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# ----------------------------- Configuration ----------------------------- #
DAGS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(DAGS_DIR, "data")
MODEL_DIR = os.path.join(DAGS_DIR, "model")
OUTPUT_DIR = os.path.join(DAGS_DIR, "working_data")

# MODIFIED: added PAYMENTS as a fourth clustering feature
FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT", "PAYMENTS"]

# MODIFIED: k = 1..15 instead of 1..49. The elbow for this data is well below 15,
# and fitting 49 models was slow without adding information.
K_RANGE = list(range(1, 16))
DEFAULT_K = 4  # fallback if no clear elbow is found

# MODIFIED: k-means++ initialisation (spreads the starting centres out) instead of "random"
KMEANS_KWARGS = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42}


def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------- Tasks ----------------------------------- #
def load_data(filename="file.csv"):
    """
    Task 1: load the raw CSV, check the required columns exist, drop rows with
    missing values, and save a clean copy.
    Returns the path to the clean CSV (a small string, safe for XCom).
    """
    _ensure_dirs()
    df = pd.read_csv(os.path.join(DATA_DIR, filename))

    # NEW: fail early with a clear message if the data does not have the expected columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    keep = (["CUST_ID"] if "CUST_ID" in df.columns else []) + FEATURES
    rows_before = len(df)
    df = df[keep].dropna(subset=FEATURES)
    print(f"Loaded {rows_before} rows; {len(df)} remain after dropping missing values")

    clean_path = os.path.join(OUTPUT_DIR, "clean_data.csv")
    df.to_csv(clean_path, index=False)
    return clean_path


def data_preprocessing(clean_data_path):
    """
    Task 2: scale the features to the 0-1 range.
    FIXED: the fitted scaler is now saved, so new customers can be scaled exactly
    the same way before prediction (the original lab predicted on unscaled data).
    Returns the path to the scaled data.
    """
    df = pd.read_csv(clean_data_path)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])

    _save_pickle(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    scaled_path = os.path.join(OUTPUT_DIR, "scaled_data.pkl")
    _save_pickle(scaled, scaled_path)
    print(f"Scaled {scaled.shape[0]} rows x {scaled.shape[1]} features")
    return scaled_path


def compute_sse(scaled_data_path):
    """
    Task 3: fit K-Means for each k in K_RANGE and record the SSE (inertia).
    Also saves the elbow curve as a CSV for inspection.
    Returns the list of SSE values.
    """
    X = _load_pickle(scaled_data_path)

    sse = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, **KMEANS_KWARGS).fit(X)
        sse.append(round(float(model.inertia_), 4))
        print(f"k={k:2d}  SSE={sse[-1]}")

    pd.DataFrame({"k": K_RANGE, "sse": sse}).to_csv(
        os.path.join(OUTPUT_DIR, "elbow_curve.csv"), index=False
    )
    return sse


def train_final_model(scaled_data_path, sse):
    """
    Task 4: find the elbow, then train and save a model with that number of clusters.
    FIXED: the original lab saved the last model from the loop (k=49) and only
    printed the elbow value. This version uses the elbow result to train the
    model that is actually saved and used for predictions.
    Returns a small dict with the chosen k and the model path.
    """
    knee = KneeLocator(K_RANGE, sse, curve="convex", direction="decreasing")
    optimal_k = int(knee.elbow) if knee.elbow is not None else DEFAULT_K
    optimal_k = max(optimal_k, 2)  # clustering into a single group is meaningless
    print(f"Optimal number of clusters (elbow method): {optimal_k}")

    X = _load_pickle(scaled_data_path)
    model = KMeans(n_clusters=optimal_k, **KMEANS_KWARGS).fit(X)

    model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    _save_pickle(model, model_path)
    return {"optimal_k": optimal_k, "model_path": model_path}


def evaluate_model(clean_data_path, scaled_data_path, model_info):
    """
    Task 5 (NEW): evaluate the clustering and describe each customer segment.
    - Silhouette score: from -1 to 1; higher means clusters are better separated.
    - Cluster profiles: the average of each feature per cluster, in original units,
      so each segment can be interpreted (e.g. "high balance, low purchases").
    Saves metrics.json and cluster_profiles.csv and returns the metrics dict.
    """
    model = _load_pickle(model_info["model_path"])
    X = _load_pickle(scaled_data_path)
    labels = model.labels_

    sil = silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42)

    df = pd.read_csv(clean_data_path)
    df["cluster"] = labels
    profiles = df.groupby("cluster")[FEATURES].mean().round(2)
    profiles["n_customers"] = df.groupby("cluster").size()
    profiles.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))
    print("Cluster profiles (feature means):")
    print(profiles.to_string())

    metrics = {
        "optimal_k": model_info["optimal_k"],
        "silhouette_score": round(float(sil), 4),
        "inertia": round(float(model.inertia_), 4),
        "cluster_sizes": {str(c): int(n) for c, n in profiles["n_customers"].items()},
        "features": FEATURES,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Silhouette score: {metrics['silhouette_score']}")
    return metrics


def predict_new_customers(model_info, filename="test.csv"):
    """
    Task 6: assign every customer in test.csv to a segment.
    FIXED: new data is scaled with the saved scaler before prediction, and all
    rows are predicted (the original returned only the first row's cluster).
    Saves predictions.csv and returns the list of cluster ids.
    """
    scaler = _load_pickle(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = _load_pickle(model_info["model_path"])

    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Test data is missing required columns: {missing}")

    X = scaler.transform(df[FEATURES])
    df["cluster"] = model.predict(X)

    df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print(df.to_string(index=False))
    return [int(c) for c in df["cluster"]]