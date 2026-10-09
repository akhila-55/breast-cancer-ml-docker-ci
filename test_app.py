
import unittest

from app import app


class TestBreastCancerApp(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(
            data["service"],
            "breast-cancer-prediction"
        )

    def test_prediction_endpoint(self):
        # Missing fields should be rejected by the API.
        response = self.client.post(
            "/predict",
            json={"mean radius": 17.99}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "Missing required fields",
            response.get_json()["error"]
        )

    def test_prediction_requires_json(self):
        response = self.client.post("/predict")

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
