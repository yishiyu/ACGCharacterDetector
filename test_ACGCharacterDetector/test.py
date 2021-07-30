import requests

resp = requests.post("http://localhost:5000/detect",
                     files={"image": open('yui.jpg','rb')})

print(resp.json())