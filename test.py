import json
import requests

# python -m simple_latex_ocr.api.run
image_path = "tests/test_files/1.png"
url = "http://127.0.0.1:8502/predict"
files = {'file': open(image_path, 'rb')}
response = requests.post(url, files=files)

# print result
data=json.loads(response.text)
print(data)