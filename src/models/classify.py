import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def prepare_classification_data(track_features: pd.DataFrame):
    df = track_features.copy()

    feature_cols = ["total_plays", "avg_minutes_played", "skip_rate", "avg_hour"]
    X = df[feature_cols].fillna(0)
    y = df["user"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder, feature_cols


def run_knn_classifier(track_features: pd.DataFrame, n_neighbors: int = 5):
    X, y, encoder, feature_cols = prepare_classification_data(track_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "accuracy": acc,
        "report": classification_report(
            y_test, y_pred, target_names=encoder.classes_
        ),
        "feature_cols": feature_cols,
        "encoder": encoder
    }


def run_decision_tree_classifier(track_features: pd.DataFrame, max_depth: int = 4):
    X, y, encoder, feature_cols = prepare_classification_data(track_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return {
        "model": model,
        "accuracy": acc,
        "report": classification_report(
            y_test, y_pred, target_names=encoder.classes_
        ),
        "feature_importances": importances,
        "feature_cols": feature_cols,
        "encoder": encoder
    }