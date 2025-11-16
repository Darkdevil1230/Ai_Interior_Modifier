import os
import sys
import argparse
import requests
import shutil
from pathlib import Path

CANDIDATE_BASE_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0/{model}",
    "https://ultralytics.com/assets/{model}",
]


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with dest_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as ex:
        print(f"  -> download failed from {url}: {ex}")
        return False


def attempt_download(model_name: str = "yolov8n.pt", out_dir: str = "weights") -> Path:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    dest = out_dir_p / model_name

    if dest.exists():
        print(f"[OK] Model already exists: {dest}")
        return dest

    print(f"[INFO] Attempting HTTP download for '{model_name}' into '{dest}'")
    for template in CANDIDATE_BASE_URLS:
        url = template.format(model=model_name)
        print(f"Trying {url} ...")
        if download_file(url, dest):
            print(f"[OK] Downloaded {model_name} to {dest}")
            return dest

    # fallback: try to let ultralytics download it (if installed)
    try:
        print("[INFO] HTTP download failed. Trying to instantiate ultralytics.YOLO to trigger auto-download...")
        from ultralytics import YOLO  # lazy import
        model = YOLO(model_name)  # this will try to download model into ultralytics cache
        # attempt to locate common cache locations
        possible_locations = [
            Path(model_name),
            Path.home() / ".cache" / "ultralytics" / model_name,
            Path.home() / ".cache" / model_name,
            Path("/root/.cache/ultralytics") / model_name,
        ]
        for p in possible_locations:
            if p.exists():
                print(f"[OK] Found cached model at {p}, copying to {dest}")
                shutil.copy2(p, dest)
                return dest
        # try attributes on model object
        src = getattr(model, "path", None) or getattr(model, "model", None)
        if isinstance(src, str) and Path(src).exists():
            Path(src).replace(dest)
            print(f"[OK] Copied model from {src} to {dest}")
            return dest
        print("[WARN] ultralytics attempted to download but script could not locate cache file automatically.")
    except Exception as e:
        print(f"[WARN] Could not use ultralytics to auto-download model: {e}")

    raise RuntimeError("Could not download YOLOv8 weights automatically; see messages above.")


def main():
    parser = argparse.ArgumentParser(description="Download YOLOv8 checkpoint and optionally save as weights/best.pt")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="Model filename to download (e.g. yolov8n.pt)")
    parser.add_argument("--out-dir", "-o", default="weights", help="Directory to save the downloaded model")
    parser.add_argument("--save-as", "-s", default=None, help="Optional: copy downloaded file to weights/<save-as> (e.g. best.pt)")
    args = parser.parse_args()

    try:
        downloaded = attempt_download(args.model, args.out_dir)
        print(f"Saved weights to: {downloaded}")
        if args.save_as:
            dest_name = Path(args.out_dir) / args.save_as
            # backup existing
            if dest_name.exists():
                bak = dest_name.with_suffix(dest_name.suffix + ".bak")
                print(f"Backing up existing {dest_name} to {bak}")
                shutil.move(str(dest_name), str(bak))
            print(f"Copying {downloaded} -> {dest_name}")
            shutil.copy2(downloaded, dest_name)
            print(f"[OK] Copied to {dest_name}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()