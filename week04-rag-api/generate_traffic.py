import time
import requests

URL = "http://3.75.219.93:8000"
API_KEY = "e2df633aa2755e20ccec39c1d29525f86824ccf6f432797af998ee660ee4f877"

questions = [
    "What is this document about?",
    "What was the revenue in 2024?",
    "Tell me about sustainability initiatives.",
]

i = 0
while True:
    # Two successful requests
    requests.post(
        f"{URL}/ask",
        json={"question": questions[i % len(questions)]},
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
    )

    # One unauthorized request
    requests.post(
        f"{URL}/ask",
        json={"question": "test"},
        headers={"Content-Type": "application/json", "X-API-Key": "wrongkey"},
    )

    # One health check
    requests.get(f"{URL}/health")

    i += 1
    print(f"Batch {i} done")
    time.sleep(30)
