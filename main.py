from src.data.load_data import load_all_data
from src.models.outliers import detect_outliers, get_top_outliers
from src.models.cluster import run_kmeans, summarize_clusters
from src.models.classify import run_knn_classifier, run_decision_tree_classifier
from src.data.clean_data import clean_spotify, clean_apple, build_combined_dataset
from src.features.build_features import add_time_features, build_track_features, build_artist_features
from src.models.similarity import run_similarity_analysis
from src.recommender.engine import recommend_songs


if __name__ == "__main__":
    datasets = load_all_data("data/raw")

    spotify_clean = clean_spotify(datasets["spotify"])
    apple_clean = clean_apple(datasets["apple_play_activity"])

    combined = build_combined_dataset(spotify_clean, apple_clean)
    combined = add_time_features(combined)

    track_features = build_track_features(combined)
    artist_features = build_artist_features(combined)
    similarity_results = run_similarity_analysis(combined, track_features, artist_features)

    print("\nSIMILARITY RESULTS:")
    for metric, value in similarity_results.items():
        print(f"{metric}: {value:.4f}")

    knn_results = run_knn_classifier(track_features, n_neighbors=5)
    tree_results = run_decision_tree_classifier(track_features, max_depth=4)
    clustered_df, kmeans_model = run_kmeans(track_features, n_clusters=3)
    cluster_summary = summarize_clusters(clustered_df)
    artist_outliers = detect_outliers(artist_features)
    top_outliers = get_top_outliers(artist_outliers, top_n=5)

    print("\nTOP OUTLIERS:")
    print(top_outliers[["user", "artist", "total_plays", "avg_minutes_played", "skip_rate"]])

    print("\nCLUSTER SUMMARY:")
    print(cluster_summary)

    print("\nKNN RESULTS:")
    print(f"Accuracy: {knn_results['accuracy']:.4f}")
    print(knn_results["report"])

    print("\nDECISION TREE RESULTS:")
    print(f"Accuracy: {tree_results['accuracy']:.4f}")
    print(tree_results["report"])

    print("\nDECISION TREE FEATURE IMPORTANCE:")
    print(tree_results["feature_importances"])

    rec_A = recommend_songs(track_features, target_user="user_A", top_n=10)
    rec_B = recommend_songs(track_features, target_user="user_B", top_n=10)

    print("\nRECOMMENDATIONS FOR USER_A:")
    print(rec_A)

    print("\nRECOMMENDATIONS FOR USER_B:")
    print(rec_B)

    print("\nCOMBINED SAMPLE:")
    print(combined.head())

    print("\nTRACK FEATURES SAMPLE:")
    print(track_features.head())

    print("\nARTIST FEATURES SAMPLE:")
    print(artist_features.head())

    print("\nTRACK FEATURES SHAPE:")
    print(track_features.shape)

    print("\nARTIST FEATURES SHAPE:")
    print(artist_features.shape)