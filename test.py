import json
import requests


image_path = "tests/test_files/5.png"
url = "http://127.0.0.1:9000/"
files = {'image': open(image_path, 'rb')}
response = requests.post(url, files=files)

# print result
print(response.text)
data=json.loads(response.text)
print("formula:\n{}".format(data['formula']))
print('confidence:\n{}'.format(data['confidence']))
print('elapse:\n{}'.format(data['elapse']))
