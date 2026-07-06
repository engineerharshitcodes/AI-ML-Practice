import requests


URL="http://peric.github.io/GetCountries/"
res=requests.get(URL)
print(res.text + "\n")
print(res.status_code)

with open("countries.html","w") as f:
    f.write(res.text)