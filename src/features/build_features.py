import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month

    def month_to_season(month):
        if month in [12, 1, 2]:
            return "winter"
        if month in [3, 4, 5]:
            return "spring"
        if month in [6, 7, 8]:
            return "summer"
        return "fall"

    df["season"] = df["month"].apply(month_to_season)
    df["minutes_played"] = df["play_duration_ms"] / 60000

    return df


def build_track_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    track_features = (
        df.groupby(["user", "platform", "track", "artist"], dropna=False)
        .agg(
            total_plays=("track", "size"),
            avg_minutes_played=("minutes_played", "mean"),
            skip_rate=("skip", "mean"),
            avg_hour=("hour", "mean"),
        )
        .reset_index()
    )

    return track_features


def build_artist_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    artist_features = (
        df.groupby(["user", "platform", "artist"], dropna=False)
        .agg(
            total_plays=("artist", "size"),
            avg_minutes_played=("minutes_played", "mean"),
            skip_rate=("skip", "mean"),
        )
        .reset_index()
    )

    return artist_features