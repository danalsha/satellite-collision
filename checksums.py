"""Write SHA-256 checksums of the raw and processed data files to data/CHECKSUMS.txt."""
import hashlib, os
from config import DATA, PROC
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
root = os.path.join(DATA, "Collision Avoidance Challenge - Dataset")
files = [os.path.join(DATA, "Collision Avoidance Challenge - Dataset.zip"),
         os.path.join(root, "raw_data", "raw_data_2015-2019.gz"),
         os.path.join(root, "kelvins_competition_data", "train_data.csv"),
         os.path.join(root, "kelvins_competition_data", "test_data.csv"),
         os.path.join(root, "kelvins_competition_data", "test_data_private.csv"),
         os.path.join(PROC, "events.csv.gz")]
with open(os.path.join(DATA, "CHECKSUMS.txt"), "w") as fh:
    for f in files:
        fh.write(f"{sha(f)}  {os.path.relpath(f, DATA)}  {os.path.getsize(f):,} bytes\n")
print(open(os.path.join(DATA, "CHECKSUMS.txt")).read())
