from pathlib import Path
import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file safely."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded: {file_path.name} -> {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path.name}: {e}")


def load_all_data(raw_data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all expected raw datasets from the raw data folder.
    """
    raw_path = Path(raw_data_dir)

    files = {
        "spotify": raw_path / "spotify_MUSIC_only (1).csv",
        "apple_container_details": raw_path / "Apple Music - Container Details (1).csv",
        "apple_track_history": raw_path / "Apple Music - Track Play History.csv",
        "apple_play_activity": raw_path / "Apple Music Play Activity.csv",
    }

    datasets = {}

    for name, file_path in files.items():
        if not file_path.exists():
            print(f"Missing file: {file_path.name}")
            continue
        datasets[name] = load_csv(file_path)

    return datasets