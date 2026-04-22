import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_outliers(artist_features: pd.DataFrame, contamination: float = 0.01):
    df = artist_features.copy()

    features = df[["total_plays", "avg_minutes_played", "skip_rate"]].fillna(0)

    model = IsolationForest(contamination=contamination, random_state=42)
    df["outlier"] = model.fit_predict(features)

    # convert: -1 = outlier, 1 = normal
    df["outlier"] = df["outlier"].apply(lambda x: 1 if x == -1 else 0)

    return df


def get_top_outliers(df: pd.DataFrame, top_n: int = 5):
    outliers = df[df["outlier"] == 1]

    top = (
        outliers.sort_values("total_plays", ascending=False)
        .groupby("user")
        .head(top_n)
    )

    return top