import requests

url = "http://127.0.0.1:5000/predict"

out = requests.post(url, files = {'image': open('../../tests/test_images/downdog_test.jpeg', 'rb')})
print(out.json())