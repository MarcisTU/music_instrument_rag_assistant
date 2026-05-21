import requests


url = "http://127.0.0.1:8006/v2/rerank"


query = "What is the capital of China?"

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "I like to eat cheese cake in china."
]

response = requests.post(url,
                         json={
                             "query": query,
                             "documents": documents,
                         }).json()

print(response)