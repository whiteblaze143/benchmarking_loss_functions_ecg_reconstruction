import urllib.request
import json
import ssl

key = "acc8fd52802842a6a18723fd828d4d6f.K-g6fwQq8RiqPsw2rRVrReHJ"

endpoints = [
    "https://api.ollama.com/v1/models",
    "https://ollama.com/v1/models",
    "https://api.ollama.com/api/tags",
    "https://ollama.com/api/tags"
]

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for url in endpoints:
    print(f"Testing {url}...")
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
            print(f"SUCCESS: {url}")
            try:
                data = json.loads(response.read().decode())
                if 'data' in data:
                    models = [m["id"] for m in data.get("data", [])]
                    print(f"Found {len(models)} models.")
                    print(models[:5])
                elif 'models' in data:
                    models = [m["name"] for m in data.get("models", [])]
                    print(f"Found {len(models)} models.")
                    print(models[:5])
            except Exception as e:
                print("Could not parse JSON:", e)
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR {e.code}: {url}")
    except Exception as e:
        print(f"ERROR: {url} - {str(e)}")
