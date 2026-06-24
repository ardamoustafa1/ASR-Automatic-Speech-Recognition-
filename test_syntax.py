import urllib.request
try:
    response = urllib.request.urlopen("http://localhost:8508")
    print(f"Server responded with: {response.getcode()}")
except Exception as e:
    print(f"Error checking server: {e}")
