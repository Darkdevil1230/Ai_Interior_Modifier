"""
Auto-download and extract the ADE20K dataset.
"""
import os
import requests
import zipfile

def download_dataset(url, save_dir="data/datasets"):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, "dataset.zip")
    print("Downloading dataset...")
    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(filename, "r") as zf:
        zf.extractall(save_dir)
    print("✅ Dataset downloaded & extracted.")

if __name__ == "__main__":
    url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    download_dataset(url)