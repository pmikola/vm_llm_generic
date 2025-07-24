import requests, pprint

URL = "http://51.79.31.174:8000"
print(requests.get(f"{URL}/health").json())

r = requests.post(f"{URL}/generate", json={"prompt": "Are you advanced model?", "max_tokens": 200})
pprint.pp(r.json())