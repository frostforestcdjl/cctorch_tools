#!/usr/bin/env python3

import argparse
import os
import re
import tempfile

from datetime import datetime, timedelta
from pathlib import Path

import fsspec


"""
Generate hourly Monterey Bay DAS raw-file lists.

Each output line:
    h5_file_path,channel

Output filename:
    mbdas_h5_raw_YYYY_JJJ_HH_00.txt

Files are uploaded to Google Cloud Storage.
"""


def get_args_parser():

    parser = argparse.ArgumentParser(
        description="Generate hourly DAS raw-file lists."
    )

    parser.add_argument(
        "--files_lst",
        type=str,
        required=True,
        help="Input text file containing DAS HDF5 paths.",
    )

    parser.add_argument(
        "--time_start",
        type=str,
        required=True,
        help="Start time (inclusive): YYYY-MM-DDTHH:MM",
    )

    parser.add_argument(
        "--time_end",
        type=str,
        required=True,
        help="End time (exclusive): YYYY-MM-DDTHH:MM",
    )

    parser.add_argument(
        "--channel_start",
        type=int,
        default=1,
        help="First channel (default: 1).",
    )

    parser.add_argument(
        "--channel_end",
        type=int,
        default=10299,
        help="Last channel, inclusive (default: 10299).",
    )

    parser.add_argument(
        "--bucket_das",
        type=str,
        default="gs://cctorch/ambient_noise_das/das_preprocess",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="mbdas_h5_raw_list",
    )

    parser.add_argument(
        "--token_file",
        type=str,
        required=True,
        help="GCS credentials JSON path or google_default.",
    )

    parser.add_argument(
        "--keep_local",
        action="store_true",
        help="Keep generated text files locally.",
    )

    parser.add_argument(
        "--local_dir",
        type=str,
        default="mbdas_h5_raw_list",
        help="Local output directory when --keep_local is used.",
    )

    return parser


def get_files_dic(files_lst):
    """
    Read HDF5 paths and group them by UTC hour.

    Example filename:
        MBARI_..._2023-10-27T174910Z.h5

    Returns:
        {
            datetime(2023, 10, 27, 17): [path1, path2, ...],
            ...
        }
    """

    files_dic = {}
    invalid_files = 0
    duplicate_files = 0
    seen = set()

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}T\d{6}Z)\.h5$",
        re.IGNORECASE,
    )

    with open(files_lst, "r") as f:

        for line in f:

            file = line.strip()

            if not file:
                continue

            match = pattern.search(Path(file).name)

            if match is None:
                invalid_files += 1
                print(f"Invalid filename: {file}")
                continue

            if file in seen:
                duplicate_files += 1
                continue

            seen.add(file)

            date_time = datetime.strptime(
                match.group(1),
                "%Y-%m-%dT%H%M%SZ",
            )

            hour = date_time.replace(
                minute=0,
                second=0,
                microsecond=0,
            )

            files_dic.setdefault(hour, []).append(file)

    for hour in files_dic:
        files_dic[hour].sort()

    print(f"Unique HDF5 files: {len(seen)}")
    print(f"Available hours: {len(files_dic)}")
    print(f"Invalid filenames: {invalid_files}")
    print(f"Duplicate paths skipped: {duplicate_files}")

    return files_dic


def get_datetime_lst(time_start, time_end):
    """
    Generate hourly timestamps.

    Start inclusive, end exclusive.
    """

    datetime_start = datetime.strptime(
        time_start,
        "%Y-%m-%dT%H:%M",
    )

    datetime_end = datetime.strptime(
        time_end,
        "%Y-%m-%dT%H:%M",
    )

    if datetime_end <= datetime_start:
        raise ValueError(
            "time_end must be later than time_start"
        )

    if (
        datetime_start.minute != 0
        or datetime_end.minute != 0
    ):
        raise ValueError(
            "time_start and time_end must be aligned "
            "to exact hours."
        )

    datetime_lst = []

    current = datetime_start

    while current < datetime_end:

        datetime_lst.append(current)

        current += timedelta(hours=1)

    return datetime_lst



def generate_raw_file_list(
    files_dic,
    datetime_lst,
    config,
    channel_start=0,
    channel_end=10299,
    keep_local=False,
    local_dir="mbdas_h5_raw_list",
):
    if channel_end < channel_start:
        raise ValueError("channel_end must be >= channel_start")

    fs = fsspec.filesystem(
        "gcs",
        token=config["token_file"],
    )

    bucket = config["bucket"].rstrip("/")
    save_path = config["save_path"].strip("/")
    destination = f"{bucket}/{save_path}"

    uploaded_count = 0
    missing_hours = []

    if keep_local:
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    for i, hour in enumerate(datetime_lst, start=1):

        datetime_str = hour.strftime("%Y_%j_%H_00")
        file_save = f"mbdas_h5_raw_{datetime_str}.txt"

        h5_files = files_dic.get(hour, [])

        if not h5_files:
            print(f"No HDF5 files for {datetime_str}")
            missing_hours.append(datetime_str)
            continue

        # Select one HDF5 file for this hour.
        # Files are sorted by filename timestamp.
        h5_file = h5_files[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            local_file = Path(temp_dir) / file_save

            with open(local_file, "w") as f:
                f.write("file_name,channel_index\n")

                for channel in range(
                    channel_start,
                    channel_end + 1,
                ):
                    f.write(f"{h5_file},{channel}\n")

            gcs_path = f"{destination}/{file_save}"

            fs.put(str(local_file), gcs_path)

            if keep_local:
                os.replace(
                    local_file,
                    Path(local_dir) / file_save,
                )

        uploaded_count += 1

        print(
            f"[{i}/{len(datetime_lst)}] Uploaded: {file_save} "
            f"| HDF5: {h5_file} "
            f"| Lines: {channel_end - channel_start + 2}"
        )

    print(f"\nUploaded files: {uploaded_count}")
    print(f"Missing hours: {len(missing_hours)}")
    print(f"Destination: {destination}")



if __name__ == "__main__":

    args = get_args_parser().parse_args()

    config = {
        "bucket": args.bucket_das,
        "save_path": args.save_path,
        "token_file": args.token_file,
    }

    files_dic = get_files_dic(
        args.files_lst
    )

    datetime_lst = get_datetime_lst(
        args.time_start,
        args.time_end,
    )

    generate_raw_file_list(
        files_dic=files_dic,
        datetime_lst=datetime_lst,
        config=config,
        channel_start=args.channel_start,
        channel_end=args.channel_end,
        keep_local=args.keep_local,
        local_dir=args.local_dir,
    )

