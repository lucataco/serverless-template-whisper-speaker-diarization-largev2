# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import base64
import requests
from io import BytesIO
import banana_dev as banana


model_payload = {
    "youtube_url":"https://www.youtube.com/watch?v=-UX0X45sYe4",
    "language": "en",
    "num_speakers": 2
}

res = requests.post("http://localhost:8000/",json=model_payload)

print(res.text)
