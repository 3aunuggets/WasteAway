import requests
url = "https://drive.google.com/file/d/1-2DLq3_riPFRkQo8g0VNJ5khQbf58J0I/view?usp=share_link"
response = requests.get(url)
open("model/model.pkl", "wb").write(response.content)

