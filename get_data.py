"""Step 0 — fetch and verify the ESA Collision Avoidance Challenge dataset.

The data is one 221 MB zip on Zenodo (record 4463683, CC BY 4.0):

    https://zenodo.org/records/4463683/files/Collision%20Avoidance%20Challenge%20-%20Dataset.zip?download=1

Run this script from the project folder.  It downloads the zip into ./data
(or uses one already there), checks its MD5 against the value Zenodo
publishes, and unpacks it.  If the download is blocked where you are, put
the zip in ./data by hand and run the script again.

    python3 get_data.py
"""
import os
import sys
import hashlib
import zipfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
URL = ("https://zenodo.org/records/4463683/files/"
       "Collision%20Avoidance%20Challenge%20-%20Dataset.zip?download=1")
ZIP = os.path.join(DATA, "Collision Avoidance Challenge - Dataset.zip")
MD5 = "d19dc8875229f2f6893253c38adddc87"        # from the Zenodo record page
os.makedirs(DATA, exist_ok=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if not os.path.exists(ZIP):
    print("downloading", URL)
    try:
        urllib.request.urlretrieve(URL, ZIP)
    except Exception as e:                      # noqa: BLE001
        sys.exit(f"download failed ({e}).  Download the zip in a browser and "
                 f"save it as:\n  {ZIP}")

got = md5(ZIP)
if got != MD5:
    sys.exit(f"MD5 mismatch: got {got}, expected {MD5}.  The file is damaged or "
             "not the Zenodo release; delete it and download again.")
print("MD5 ok:", got, f"({os.path.getsize(ZIP):,} bytes)")

with zipfile.ZipFile(ZIP) as z:
    for info in z.infolist():
        print(f"  {info.filename}  {info.file_size:,} bytes")
    z.extractall(DATA)
print("unpacked into", DATA)
