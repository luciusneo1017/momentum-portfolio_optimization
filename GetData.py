import requests
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=IBM&apikey=LWB63L7XZ16BSFW5'
r = requests.get(url)
data=r.json()
print(data)