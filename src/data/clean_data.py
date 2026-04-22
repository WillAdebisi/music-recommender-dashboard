import pandas as pd


def clean_spotify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={
        "ts": "timestamp",
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "ms_played": "play_duration_ms",
        "skipped": "skip"
    })

    df = df[[
        "timestamp",
        "track",
        "artist",
        "play_duration_ms",
        "skip"
    ]]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["platform"] = "spotify"
    df["user"] = "user_A"

    df = df.dropna(subset=["track", "artist", "timestamp"])
    return df


def clean_apple(play_activity: pd.DataFrame) -> pd.DataFrame:
    df = play_activity.copy()

    df = df.rename(columns={
        "Song Name": "track",
        "Play Duration Milliseconds": "play_duration_ms",
        "Event Start Timestamp": "timestamp"
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Extract artist from "Artist - Song"
    df["artist"] = df["track"].str.split(" - ").str[0]
    df["track"] = df["track"].str.split(" - ").str[-1]

    df = df[[
        "timestamp",
        "track",
        "artist",
        "play_duration_ms"
    ]]

    df["skip"] = df["play_duration_ms"].fillna(0) < 30000
    df["platform"] = "apple"
    df["user"] = "user_B"

    df = df.dropna(subset=["track", "artist", "timestamp"])

    return df

def build_combined_dataset(spotify_df: pd.DataFrame, apple_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([spotify_df, apple_df], ignore_index=True)
    return combined