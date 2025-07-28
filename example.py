import requests, pprint
from config import *

URL = f"http://{HOST}:{PORT}"
print(requests.get(f"{URL}/health").json())
r = requests.post(f"{URL}/generate", json={"prompt": "Could you introduce yourself?", "max_tokens": 200})
pprint.pp(r.json())