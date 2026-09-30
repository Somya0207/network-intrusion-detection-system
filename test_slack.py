import requests
import os
from dotenv import load_dotenv

load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

message = {"text": "🚨 Test alert from NIDS project - Slack integration working."}
response = requests.post(SLACK_WEBHOOK_URL, json=message, timeout=5)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")