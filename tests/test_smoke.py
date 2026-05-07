import unittest

from app import app


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_landing_page_is_reachable(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_login_page_is_reachable(self):
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
