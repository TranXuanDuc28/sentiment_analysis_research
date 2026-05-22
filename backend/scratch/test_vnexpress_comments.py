import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "https://usi-saas.vnexpress.net/index/get?objectid=5074297&objecttype=3&siteid=1006219&limit=24"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=10)
print("Status code:", response.status_code)
data = response.json()
if "data" in data and "items" in data["data"]:
    items = data["data"]["items"]
    print(f"Found {len(items)} comments!")
    for idx, item in enumerate(items[:10]):
         print(f"{idx+1}. User: {item.get('full_name')} | Comment: {item.get('content')}")
else:
    print("Failed. Response:", data)
