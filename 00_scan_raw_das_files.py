import argparse
import fsspec
from datetime import datetime

# --- command-line arguments ---
parser = argparse.ArgumentParser(
    description="Scan available DAS HDF5 files in Google Cloud Storage."
)

parser.add_argument(
    "--token_file",
    type=str,
    default="",
    help="Path to Google Cloud credentials JSON file. "
         "Default: google_default"
)

args = parser.parse_args()

# --- configure credentials ---
token_file = args.token_file

if not token_file:
    print("No token file provided. Using default Google Cloud credentials.")
    token_file = "google_default"
else:
    print(f"Using provided token file: {token_file}")

# --- scan h5 files ---
print("Start scan available DAS files...")

fs = fsspec.filesystem("gcs", token=token_file)

files = fs.find("berkeley-mbari-das")
print(f"Found {len(files)} files in the bucket")

h5_files = sorted(f for f in files if f.endswith(".h5"))
print(f"Found {len(h5_files)} h5 files")

# --- save DAS file list ---
date = datetime.now().strftime("%Y%m%d")
filename = f"mbdas_h5_file_list_until_{date}.txt"

with open(filename, "w") as f:
    for file in h5_files:
        f.write(f"{file}\n")

print(f"Done. DAS file list is saved to {filename}")
