import json

from django.test import Client, TestCase


class ChatAPITestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_chat_endpoint_works(self):
        response = self.client.post(
            '/api/chat/',  # Change if your path is different
            json.dumps({'query': "Give the facts in the 2009 case on bounced checks and estafa."}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        print("Chatbot reply:", data["response"])
