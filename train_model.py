from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODEL_PATH = Path("breast_cancer_model.pkl")


def train_model():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save feature names so the application can identify the input columns.
    model.feature_names_in_ = data.feature_names

    joblib.dump(model, MODEL_PATH)

    accuracy = model.score(X_test, y_test)

    print("Breast Cancer model trained successfully.")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
