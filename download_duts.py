"""
download_duts.py

Utility script to download and unpack the DUTS Salient Object Detection dataset.

Steps to use:

1. Go to the official DUTS dataset page in your browser
   (search: "DUTS saliency dataset") and copy the direct download links
   for:
      - DUTS-TR (training set .zip)
      - DUTS-TE (test set .zip)

2. Paste those URLs below in TRAIN_ZIP_URL and TEST_ZIP_URL.

3. Run this script from the project root:

       python download_duts.py

4. After it finishes, all images will be placed in:

       data/DUTS/images/
       data/DUTS/masks/

   which matches data_loader.py.
"""

import os
import zipfile
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from tqdm import tqdm
 
print("DEBUG: download_duts.py is running!")


# TODO: paste the real URLs here from the DUTS website
TRAIN_ZIP_URL = "http://saliencydetection.net/duts/download/DUTS-TR.zip"
TEST_ZIP_URL  = "http://saliencydetection.net/duts/download/DUTS-TE.zip"



def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def download_file(url: str, dest_path: str):
    """
    Download a file from `url` to `dest_path` with a simple progress bar.
    """
    print(f"\nDownloading:\n  {url}\n→ {dest_path}")

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as response:
            total_size = int(response.info().get("Content-Length", -1))
            chunk_size = 1024 * 1024  # 1 MB

            with open(dest_path, "wb") as f, tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(dest_path),
            ) as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    if total_size > 0:
                        pbar.update(len(chunk))

    except HTTPError as e:
        print(f"HTTP error while downloading {url}: {e}")
        raise
    except URLError as e:
        print(f"URL error while downloading {url}: {e}")
        raise


def extract_zip(zip_path: str, extract_to: str):
    print(f"\nExtracting {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def collect_images_and_masks(raw_root: str, images_dest: str, masks_dest: str):
    """
    Walk through `raw_root` and copy all files from folders whose names end
    with 'Image' into images_dest, and those ending with 'Mask' into masks_dest.

    This is designed to work with typical DUTS structure like:
      DUTS-TR/DUTS-TR-Image
      DUTS-TR/DUTS-TR-Mask
      DUTS-TE/DUTS-TE-Image
      DUTS-TE/DUTS-TE-Mask
    """
    import shutil

    image_dirs = []
    mask_dirs = []

    for root, dirs, files in os.walk(raw_root):
        for d in dirs:
            full = os.path.join(root, d)
            name = d.lower()
            if name.endswith("image"):
                image_dirs.append(full)
            elif name.endswith("mask"):
                mask_dirs.append(full)

    print(f"\nFound image dirs: {image_dirs}")
    print(f"Found mask dirs:  {mask_dirs}")

    safe_mkdir(images_dest)
    safe_mkdir(masks_dest)

    # Copy images
    for d in image_dirs:
        for fname in os.listdir(d):
            src = os.path.join(d, fname)
            if os.path.isfile(src):
                dst = os.path.join(images_dest, fname)
                shutil.copy2(src, dst)

    # Copy masks
    for d in mask_dirs:
        for fname in os.listdir(d):
            src = os.path.join(d, fname)
            if os.path.isfile(src):
                dst = os.path.join(masks_dest, fname)
                shutil.copy2(src, dst)

    print(f"\nCopied all images to {images_dest}")
    print(f"Copied all masks  to {masks_dest}")


def main():
    # Paths inside your project
    data_root = os.path.join("data", "DUTS")
    raw_root = os.path.join(data_root, "raw")
    zips_root = os.path.join(data_root, "zips")
    images_dest = os.path.join(data_root, "images")
    masks_dest = os.path.join(data_root, "masks")

    safe_mkdir(data_root)
    safe_mkdir(raw_root)
    safe_mkdir(zips_root)

    train_zip_path = os.path.join(zips_root, "DUTS-TR.zip")
    test_zip_path = os.path.join(zips_root, "DUTS-TE.zip")

    if "PASTE_DUTS_TR_ZIP_URL_HERE" in TRAIN_ZIP_URL or \
       "PASTE_DUTS_TE_ZIP_URL_HERE" in TEST_ZIP_URL:
        print(
            "ERROR: Please edit download_duts.py and paste the real DUTS URLs "
            "into TRAIN_ZIP_URL and TEST_ZIP_URL before running."
        )
        return

    # Download if not already present
    if not os.path.exists(train_zip_path):
        download_file(TRAIN_ZIP_URL, train_zip_path)
    else:
        print(f"Skipping download, found existing: {train_zip_path}")

    if not os.path.exists(test_zip_path):
        download_file(TEST_ZIP_URL, test_zip_path)
    else:
        print(f"Skipping download, found existing: {test_zip_path}")

    # Extract both zips into raw_root
    extract_zip(train_zip_path, raw_root)
    extract_zip(test_zip_path, raw_root)

    # Collect all images/masks into unified folders
    collect_images_and_masks(raw_root, images_dest, masks_dest)

    print("\n✅ DUTS download and extraction complete.")
    print(f"Images in: {images_dest}")
    print(f"Masks in:  {masks_dest}")


if __name__ == "__main__":
    main()
