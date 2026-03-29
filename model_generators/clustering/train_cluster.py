from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / 'dummy-data' / 'vehicles_ml_dataset.csv'
MODEL_PATH = BASE_DIR / 'model_generators' / 'clustering' / 'clustering_model.pkl'
SCALER_PATH = BASE_DIR / 'model_generators' / 'clustering' / 'clustering_scaler.pkl'
MAPPING_PATH = BASE_DIR / 'model_generators' / 'clustering' / 'cluster_mapping.pkl'

SEGMENT_FEATURES = ['estimated_income', 'selling_price']


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    if mean_val == 0:
        return 0.0
    return round((std_val / mean_val) * 100, 2)


def train_and_save_clustering_model() -> dict:
    df = pd.read_csv(DATASET_PATH)
    X = df[SEGMENT_FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_kmeans = None
    best_score = -1.0

    for k in range(2, 5):
        candidate = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = candidate.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_kmeans = candidate

    labels = best_kmeans.predict(X_scaled)
    df['cluster_id'] = labels

    centers_unscaled = scaler.inverse_transform(best_kmeans.cluster_centers_)
    sorted_clusters = centers_unscaled[:, 0].argsort()
    names = ['Economy', 'Standard', 'Premium', 'Elite']
    cluster_mapping = {
        cluster_id: names[idx]
        for idx, cluster_id in enumerate(sorted_clusters)
    }

    df['client_class'] = df['cluster_id'].map(cluster_mapping)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(cluster_mapping, MAPPING_PATH)

    silhouette_avg = round(best_score, 4)
    variation = _coefficient_of_variation(df['selling_price'].to_numpy())

    cluster_summary = df.groupby('client_class')[SEGMENT_FEATURES].mean().reset_index()
    cluster_counts = df['client_class'].value_counts().reset_index()
    cluster_counts.columns = ['client_class', 'count']
    cluster_summary = cluster_summary.merge(cluster_counts, on='client_class')

    comparison_df = df[['client_name', 'district', 'estimated_income', 'selling_price', 'client_class']]

    return {
        'silhouette': silhouette_avg,
        'coefficient_variation': variation,
        'summary_df': cluster_summary,
        'comparison_df': comparison_df,
        'n_clusters': int(best_kmeans.n_clusters),
    }


def evaluate_clustering_model() -> dict:
    metrics = train_and_save_clustering_model()
    return {
        'silhouette': metrics['silhouette'],
        'coefficient_variation': metrics['coefficient_variation'],
        'n_clusters': metrics['n_clusters'],
        'summary': metrics['summary_df'].to_html(
            classes='table table-bordered table-striped table-sm',
            float_format='%.2f',
            justify='center',
            index=False,
        ),
        'comparison': metrics['comparison_df'].head(10).to_html(
            classes='table table-bordered table-striped table-sm',
            float_format='%.2f',
            justify='center',
            index=False,
        ),
    }
