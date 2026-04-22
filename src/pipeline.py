from src.data.load_data import load_all_data
from src.data.clean_data import clean_spotify, clean_apple, build_combined_dataset
from src.features.build_features import add_time_features, build_track_features, build_artist_features
from src.models.similarity import run_similarity_analysis
from src.models.classify import run_knn_classifier, run_decision_tree_classifier
from src.models.cluster import run_kmeans, summarize_clusters
from src.models.outliers import detect_outliers, get_top_outliers
from src.recommender.engine import recommend_songs


def run_full_pipeline(raw_data_dir: str = "data/raw") -> dict:
    datasets = load_all_data(raw_data_dir)

    spotify_clean = clean_spotify(datasets["spotify"])
    apple_clean = clean_apple(datasets["apple_play_activity"])

    combined = build_combined_dataset(spotify_clean, apple_clean)
    combined = add_time_features(combined)

    track_features = build_track_features(combined)
    artist_features = build_artist_features(combined)

    similarity_results = run_similarity_analysis(combined, track_features, artist_features)

    knn_results = run_knn_classifier(track_features, n_neighbors=5)
    tree_results = run_decision_tree_classifier(track_features, max_depth=4)

    clustered_df, _ = run_kmeans(track_features, n_clusters=3)
    cluster_summary = summarize_clusters(clustered_df)

    artist_outliers = detect_outliers(artist_features)
    top_outliers = get_top_outliers(artist_outliers, top_n=5)

    rec_a = recommend_songs(track_features, target_user="user_A", top_n=10)
    rec_b = recommend_songs(track_features, target_user="user_B", top_n=10)

    return {
        "datasets": datasets,
        "combined": combined,
        "track_features": track_features,
        "artist_features": artist_features,
        "similarity_results": similarity_results,
        "knn_results": knn_results,
        "tree_results": tree_results,
        "clustered_df": clustered_df,
        "cluster_summary": cluster_summary,
        "artist_outliers": artist_outliers,
        "top_outliers": top_outliers,
        "rec_a": rec_a,
        "rec_b": rec_b,
    }