import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.pipeline import run_full_pipeline

st.set_page_config(
    page_title="Music Recommender Dashboard",
    page_icon="🎵",
    layout="wide"
)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, target_path: Path) -> None:
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def clear_raw_dir():
    for item in RAW_DIR.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def plot_similarity(similarity_results: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(similarity_results.keys())
    values = list(similarity_results.values())
    ax.bar(keys, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Similarity Metrics")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)


def plot_cluster_summary(cluster_summary: pd.DataFrame):
    pivot_counts = cluster_summary.pivot(index="cluster", columns="user", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot_counts.plot(kind="bar", ax=ax)
    ax.set_title("Cluster Counts by User")
    ax.set_ylabel("Track Count")
    ax.set_xlabel("Cluster")
    st.pyplot(fig)


def plot_feature_importance(importances: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 4))
    importances.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Decision Tree Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)


st.title("🎵 Personalized Music Recommender Dashboard")
st.caption("Upload Spotify and Apple Music data, compare two users, and generate cross-platform recommendations.")

with st.sidebar:
    st.header("Upload data")
    spotify_file = st.file_uploader("Spotify CSV", type=["csv"], key="spotify")
    apple_activity = st.file_uploader("Apple Music Play Activity CSV", type=["csv"], key="apple_activity")
    apple_history = st.file_uploader("Apple Track Play History CSV", type=["csv"], key="apple_history")
    apple_container = st.file_uploader("Apple Container Details CSV", type=["csv"], key="apple_container")

    run_button = st.button("Run analysis", type="primary")
    reset_button = st.button("Clear uploaded files")

if reset_button:
    clear_raw_dir()
    st.success("Cleared data/raw.")

if run_button:
    missing = []
    if spotify_file is None:
        missing.append("Spotify CSV")
    if apple_activity is None:
        missing.append("Apple Music Play Activity CSV")
    if apple_history is None:
        missing.append("Apple Track Play History CSV")
    if apple_container is None:
        missing.append("Apple Container Details CSV")

    if missing:
        st.error("Missing files: " + ", ".join(missing))
        st.stop()

    clear_raw_dir()

    save_uploaded_file(spotify_file, RAW_DIR / "spotify_MUSIC_only (1).csv")
    save_uploaded_file(apple_activity, RAW_DIR / "Apple Music Play Activity.csv")
    save_uploaded_file(apple_history, RAW_DIR / "Apple Music - Track Play History.csv")
    save_uploaded_file(apple_container, RAW_DIR / "Apple Music - Container Details (1).csv")

    with st.spinner("Running full pipeline..."):
        results = run_full_pipeline(str(RAW_DIR))

    st.success("Analysis complete.")

    similarity_results = results["similarity_results"]
    cluster_summary = results["cluster_summary"]
    top_outliers = results["top_outliers"]
    rec_a = results["rec_a"]
    rec_b = results["rec_b"]
    knn_results = results["knn_results"]
    tree_results = results["tree_results"]
    combined = results["combined"]
    track_features = results["track_features"]
    artist_features = results["artist_features"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hour Similarity", f"{similarity_results['hour_similarity']:.4f}")
    c2.metric("Season Similarity", f"{similarity_results['season_similarity']:.4f}")
    c3.metric("Artist Overlap Top 20", f"{similarity_results['artist_overlap_top20']:.4f}")
    c4.metric("Behavior Similarity", f"{similarity_results['behavior_similarity']:.4f}")

    st.subheader("Similarity Overview")
    plot_similarity(similarity_results)

    left, right = st.columns(2)

    with left:
        st.subheader("Classification")
        st.write(f"**KNN Accuracy:** {knn_results['accuracy']:.4f}")
        st.code(knn_results["report"])

    with right:
        st.subheader("Decision Tree")
        st.write(f"**Decision Tree Accuracy:** {tree_results['accuracy']:.4f}")
        st.code(tree_results["report"])
        plot_feature_importance(tree_results["feature_importances"])

    st.subheader("Clustering")
    st.dataframe(cluster_summary, use_container_width=True)
    plot_cluster_summary(cluster_summary)

    st.subheader("Top Outliers")
    st.dataframe(
        top_outliers[["user", "artist", "total_plays", "avg_minutes_played", "skip_rate"]],
        use_container_width=True
    )

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.subheader("Recommendations for User A")
        st.dataframe(rec_a, use_container_width=True)

    with rec_col2:
        st.subheader("Recommendations for User B")
        st.dataframe(rec_b, use_container_width=True)

    with st.expander("Preview processed data"):
        st.write("Combined shape:", combined.shape)
        st.dataframe(combined.head(50), use_container_width=True)

        st.write("Track features shape:", track_features.shape)
        st.dataframe(track_features.head(50), use_container_width=True)

        st.write("Artist features shape:", artist_features.shape)
        st.dataframe(artist_features.head(50), use_container_width=True)

    csv_a = rec_a.to_csv(index=False).encode("utf-8")
    csv_b = rec_b.to_csv(index=False).encode("utf-8")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download User A Recommendations",
        data=csv_a,
        file_name="recommendations_user_a.csv",
        mime="text/csv"
    )
    d2.download_button(
        "Download User B Recommendations",
        data=csv_b,
        file_name="recommendations_user_b.csv",
        mime="text/csv"
    )

else:
    st.info("Upload the four CSV files in the sidebar, then click Run analysis.")