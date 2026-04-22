import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_from_series(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).fillna(0)
    if aligned.shape[0] == 0:
        return 0.0
    return float(
        cosine_similarity(
            aligned.iloc[:, 0].values.reshape(1, -1),
            aligned.iloc[:, 1].values.reshape(1, -1)
        )[0][0]
    )


def hour_similarity(combined_df: pd.DataFrame) -> float:
    hour_dist = (
        combined_df.groupby(["user", "hour"])
        .size()
        .unstack(fill_value=0)
        .T
    )

    # normalize to proportions
    hour_dist = hour_dist.div(hour_dist.sum(axis=0), axis=1)

    if "user_A" not in hour_dist.columns or "user_B" not in hour_dist.columns:
        return 0.0

    return cosine_from_series(hour_dist["user_A"], hour_dist["user_B"])


def season_similarity(combined_df: pd.DataFrame) -> float:
    season_dist = (
        combined_df.groupby(["user", "season"])
        .size()
        .unstack(fill_value=0)
        .T
    )

    season_dist = season_dist.div(season_dist.sum(axis=0), axis=1)

    if "user_A" not in season_dist.columns or "user_B" not in season_dist.columns:
        return 0.0

    return cosine_from_series(season_dist["user_A"], season_dist["user_B"])


def artist_overlap(artist_features: pd.DataFrame, top_n: int = 20) -> float:
    top_artists = (
        artist_features.sort_values(["user", "total_plays"], ascending=[True, False])
        .groupby("user")
        .head(top_n)
    )

    user_a = set(top_artists[top_artists["user"] == "user_A"]["artist"].dropna())
    user_b = set(top_artists[top_artists["user"] == "user_B"]["artist"].dropna())

    if not user_a and not user_b:
        return 0.0

    union = user_a.union(user_b)
    intersection = user_a.intersection(user_b)

    return len(intersection) / len(union) if union else 0.0


def behavior_similarity(track_features: pd.DataFrame) -> float:
    summary = (
        track_features.groupby("user")
        .agg(
            avg_minutes_played=("avg_minutes_played", "mean"),
            skip_rate=("skip_rate", "mean")
        )
    )

    if "user_A" not in summary.index or "user_B" not in summary.index:
        return 0.0

    a = summary.loc["user_A"]
    b = summary.loc["user_B"]

    # convert distance to similarity
    distance = np.sqrt(
        (a["avg_minutes_played"] - b["avg_minutes_played"]) ** 2 +
        (a["skip_rate"] - b["skip_rate"]) ** 2
    )

    return 1 / (1 + distance)


def run_similarity_analysis(combined_df: pd.DataFrame, track_features: pd.DataFrame, artist_features: pd.DataFrame) -> dict:
    results = {
        "hour_similarity": hour_similarity(combined_df),
        "season_similarity": season_similarity(combined_df),
        "artist_overlap_top20": artist_overlap(artist_features, top_n=20),
        "behavior_similarity": behavior_similarity(track_features),
    }
    return results