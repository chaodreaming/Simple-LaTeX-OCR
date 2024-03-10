import json
import requests


image_path = "tests/test_files/1.png"
url = "http://127.0.0.1:5000/upload"
files = {'file': open(image_path, 'rb')}
response = requests.post(url, files=files)

# print result
print(response.text)
data=json.loads(response.text)["latex"]
print("formula:\n{}".format(data['formula']))
print('confidence:\n{}'.format(data['confidence']))
print('elapse:\n{}'.format(data['elapse']))
