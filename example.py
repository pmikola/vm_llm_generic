import requests, pprint

URL = "http://51.79.26.54:8000"
print(requests.get(f"{URL}/health").json())
r = requests.post(f"{URL}/generate", json={"prompt": "Are you insane?", "max_tokens": 128})
pprint.pp(r.json())