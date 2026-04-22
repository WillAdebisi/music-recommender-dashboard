import pandas as pd


def recommend_songs(track_features: pd.DataFrame, target_user: str = "user_A", top_n: int = 10):
    other_user = "user_B" if target_user == "user_A" else "user_A"

    target_df = track_features[track_features["user"] == target_user]
    candidate_df = track_features[track_features["user"] == other_user].copy()

    # user behavior profile
    avg_minutes = target_df["avg_minutes_played"].mean()
    avg_skip = target_df["skip_rate"].mean()

    # similarity scores
    candidate_df["minutes_score"] = 1 / (1 + abs(candidate_df["avg_minutes_played"] - avg_minutes))
    candidate_df["skip_similarity"] = 1 / (1 + abs(candidate_df["skip_rate"] - avg_skip))

    # HARD penalty for bad songs
    candidate_df["skip_penalty"] = candidate_df["skip_rate"]

    # final score (skip dominates)
    candidate_df["score"] = (
        0.5 * candidate_df["skip_similarity"]
        + 0.3 * candidate_df["minutes_score"]
        - 0.4 * candidate_df["skip_penalty"]
    )

    # remove terrible songs completely
    candidate_df = candidate_df[candidate_df["skip_rate"] < 0.5]

    recommendations = (
        candidate_df.sort_values("score", ascending=False)
        .drop_duplicates(subset=["track", "artist"])
        .head(top_n)
    )

    return recommendations[
        ["track", "artist", "score", "total_plays", "avg_minutes_played", "skip_rate"]
    ]