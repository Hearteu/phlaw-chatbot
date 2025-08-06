import json

from django.test import Client, TestCase


class ChatAPITestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_chat_endpoint_works(self):
        response = self.client.post(
            '/api/chat/',  # Change if your path is different
            json.dumps({'query': "Give me the ruling of the G.R. No. 252898 case."}),
            # json.dumps({'query': "What is the issue in the G.R. No. 252898 case."}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        print("Chatbot reply:", data["response"])
