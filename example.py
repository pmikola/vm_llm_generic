import requests, pprint
from config import *

URL = f"http://{HOST}:{PORT}"
print(requests.get(f"{URL}/health").json())
r = requests.post(f"{URL}/generate", json={"prompt": "Are you insane?", "max_tokens": 128})
pprint.pp(r.json())