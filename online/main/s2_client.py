# user调client
import requests
import json

data = {
    "userid": 1
}

response = requests.post(
    "http://localhost:5000/ml/rec",
    data=json.dumps(data))
print(response.text)
