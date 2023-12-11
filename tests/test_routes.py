import unittest
import json
from flask import Flask, jsonify
from app import app  # replace 'your_flask_app' with the name of your flask app

class FlaskTest(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True 

    def test_read_root(self):
        response = self.app.get('/')
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, {"Status": "OK"})
         
    def test_return_topics(self):
        response = self.app.get('/topics')
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data, dict)  

    def test_return_popularity(self):
        response = self.app.get('/topic_popularity/?topic_id=1')  # assuming topic_id 1 exists
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data, dict)  # assuming topic_popularity_dict[1] is a dictionary

    def test_return_sentiment(self):
        response = self.app.get('/topic_sentiment/?topic_id=1')  # assuming topic_id 1 exists
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data, dict)  # assuming topic_sentiment_dict[1] is a dictionary

    def test_predict_topic(self):
        response = self.app.post('/predict_topic/', json={'text': 'This is a test sentence.'})
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data, dict)  # assuming the response is a dictionary

    def test_get_documents(self):
        response = self.app.get('/get_documents/?topic_id=1&threshold=0.5')  # assuming topic_id 1 exists and threshold is 0.5
        data = json.loads(response.get_data())
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(data, dict)  # assuming the response is a dictionary

if __name__ == "__main__":
    unittest.main()