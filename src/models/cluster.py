import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_kmeans(track_features: pd.DataFrame, n_clusters: int = 3):
    df = track_features.copy()

    features = df[["avg_minutes_played", "skip_rate"]].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    return df, kmeans


def summarize_clusters(clustered_df: pd.DataFrame):
    summary = (
        clustered_df.groupby(["cluster", "user"])
        .agg(
            count=("track", "size"),
            avg_minutes=("avg_minutes_played", "mean"),
            avg_skip=("skip_rate", "mean")
        )
        .reset_index()
    )

    return summary