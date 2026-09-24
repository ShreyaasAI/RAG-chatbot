from dotenv import load_dotenv
import os
import requests

load_dotenv()
API = os.getenv("QDRANT_API_KEY")
headers={
    "api-key": API,
    "Content-type":"application/json"
}
response = requests.get(
    os.getenv("QDRANT_CLUSTER_ENDPOINT"),
    headers = headers
)
print(response.status_code)
print(response.json())