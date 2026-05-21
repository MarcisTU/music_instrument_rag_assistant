import numpy as np
import requests


url = "http://127.0.0.1:8004/v1/embeddings"

input = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "I like to eat cheese cake in china."
]

response = requests.post(url,
                         json={
                             "input": input,
                         }).json()

# print first embedding
print(np.array(response["data"][0]["embedding"]).shape)