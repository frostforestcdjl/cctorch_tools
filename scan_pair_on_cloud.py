import argparse
import zarr
import gcsfs
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Extract station pairs from GCS Zarr files")
parser.add_argument("--project_layer", type=int, choices=[0, 1], required=True, help="0: ccf_project, 1: ccf_project_folder/project")
parser.add_argument("--project_folder", type=str, default=None, help="Project folder, e.g., M5eq_run2 (required when project_layer=1)")
parser.add_argument("--project", type=str, required=True, help="Project name, e.g., M5eq_1")
args = parser.parse_args()

project_layer = args.project_layer
project_folder = args.project_folder
project = args.project

# Validate arguments
if project_layer == 1 and not project_folder:
    parser.error("--project_folder is required when --project_layer=1")

# Initialize GCS
fs = gcsfs.GCSFileSystem(asynchronous=False, skip_instance_cache=True)

# Construct GCS path
if project_layer == 0:
    gcs_path = f"gs://cctorch/ambient_noise/ccf_{project}"
else:
    gcs_path = f"gs://cctorch/ambient_noise/ccf_{project_folder}/{project}"

print(f"GCS path: {gcs_path}")

file_save = f"station_pair_{project}.txt"
log_save = f"station_pair_{project}_log.txt"

folder_year = fs.ls(gcs_path)

pair_set = set()
error_list = []

for year in folder_year:
    files = fs.ls(year)
    print(f"{len(files)} days in {year}")

    for file in files:
        print(f"Loading {file}...")

        store = fs.get_mapper(f"gs://{file}")

        try:
            z = zarr.open(store, mode="r")
    
            id1_list = z["id1"][:]
            id2_list = z["id2"][:]
    
            for id1, id2 in zip(id1_list, id2_list):
                pair_set.add(f"{id1}_{id2}")
        except Exception as e:
            error_msg = f"{file}\t{type(e).__name__}: {e}"
            print(f"Error processing: {error_msg}")
            error_list.append(error_msg)
            

# Sort unique pairs
pair_list = sorted(pair_set)

# Convert to network.station format
pair_list_mwcs = sorted({
    "_".join(
        ".".join(station.split(".")[:2])
        for station in pair.split("_")
    )
    for pair in pair_list
})

# Create output directory
output_dir = Path("txt")
output_dir.mkdir(parents=True, exist_ok=True)

output_log_dir = Path("logs")
output_log_dir.mkdir(parents=True, exist_ok=True)

# Save original station pairs
file_path = output_dir / file_save
file_path.write_text("\n".join(pair_list))

# Save MWCS station pairs
file_path_mwcs = output_dir / f"station_pair_mwcs_{project}.txt"
file_path_mwcs.write_text("\n".join(pair_list_mwcs))

# Save error days
file_path_log = output_log_dir / log_save
log_content = [
    f"Project: {project}",
    f"Original pairs: {len(pair_list)}",
    f"MWCS pairs: {len(pair_list_mwcs)}",
    f"Errors: {len(error_list)}",
    "",
    "Failed paths:",
    *error_list,
]
file_path_log.write_text("\n".join(log_content) + "\n")

print(f"Original pairs: {len(pair_list)}")
print(f"MWCS pairs: {len(pair_list_mwcs)}")

print(f"Saved: {file_path.resolve()}")
print(f"Saved: {file_path_mwcs.resolve()}")

print(f"Errors: {len(error_list)}")
print(f"Log saved: {file_path_log.resolve()}")
