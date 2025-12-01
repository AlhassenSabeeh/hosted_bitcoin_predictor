# download_latest_model.py

import os
import requests
from pathlib import Path

REPO = "Tbaosman/bitcoin_predictor" 
ASSET_PREFIX = "bitcoin_model"    # your artifact prefix
OUT_DIR = Path("models/saved_models")
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"

TOKEN = os.environ.get("GITHUB_TOKEN")

def download(url, filename):
    headers = {"Accept": "application/octet-stream"}
    if TOKEN:
        headers["Authorization"] = f"token {TOKEN}"

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {"Accept": "application/vnd.github+json"}
    if TOKEN:
        headers["Authorization"] = f"token {TOKEN}"

    r = requests.get(API_URL, headers=headers)
    r.raise_for_status()
    release = r.json()

    assets = release.get("assets", [])
    if not assets:
        print("No release assets found!")
        return

    for asset in assets:
        name = asset["name"]
        if name.startswith(ASSET_PREFIX):
            print("Downloading:", name)
            url = asset["browser_download_url"]
            download(url, OUT_DIR / name)
            print("Saved:", OUT_DIR / name)

if __name__ == "__main__":
    main()
