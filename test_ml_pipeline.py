
import os
import unittest

import joblib
from sklearn.datasets import load_breast_cancer


class TestBreastCancerPipeline(unittest.TestCase):

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists("breast_cancer_model.pkl"),
            "Trained model file was not found"
        )

    def test_model_predicts_valid_classes(self):
        model = joblib.load("breast_cancer_model.pkl")
        data = load_breast_cancer()

        predictions = model.predict(data.data[:5])

        self.assertEqual(len(predictions), 5)
        self.assertTrue(
            set(predictions).issubset({0, 1}),
            "Model returned unexpected class values"
        )

    def test_model_has_30_features(self):
        model = joblib.load("breast_cancer_model.pkl")

        self.assertEqual(len(model.feature_names_in_), 30)


if __name__ == "__main__":
    unittest.main()
