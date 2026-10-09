import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


MODEL_PATH = "breast_cancer_model.pkl"
MIN_ACCURACY = 0.90


def main():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Required accuracy: {MIN_ACCURACY:.4f}")

    if accuracy < MIN_ACCURACY:
        print("QUALITY GATE: FAIL")
        raise SystemExit(1)

    print("QUALITY GATE: PASS")


if __name__ == "__main__":
    main()
