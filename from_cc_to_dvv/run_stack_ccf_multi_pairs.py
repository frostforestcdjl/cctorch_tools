import os
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"

import numpy as np
import pandas as pd
import zarr
from obspy.signal.filter import bandpass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
from gcsfs.core import GCSFileSystem

from args import parse_args

import multiprocessing as mp


def get_pairs_from_file(station_pair_list):
    print(f"Load pairs from {station_pair_list}")

    pairs_lst = []
    with open(station_pair_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pairs = line.split(' ')[0]
            pairs = pairs.strip()
            st1, st2 = pairs.split('_')
            st1_nw_sta = f"{st1.split('.')[0]}.{st1.split('.')[1]}"
            st2_nw_sta = f"{st2.split('.')[0]}.{st2.split('.')[1]}"
            pairs_lst.append([st1_nw_sta, st2_nw_sta])

    print(f"{len(pairs_lst)} pairs in the list in total.")
    return pairs_lst

def log_progress(pair, folder_stacks, filterid, component, log_type=""):
    print(f"Finish and log progress for pair {pair} at filter {filterid:02d} and component {component}")
    fs_local = GCSFileSystem()
    filename_log = os.path.join(folder_stacks, f"logs/{filterid:02d}/stack_log_{filterid:02d}_{component}{log_type}.txt")
    try:
        with fs_local.open(filename_log, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"No existing log file found at {filename_log}. Starting fresh.")
        content = []

    if content:
        print(f"Existing log content at {filename_log}: {content}")
        content = content.strip().split('\n')
        content = list(set(content))  # Remove duplicates
        content.sort()  # Sort for consistency

    content += [f"{pair[0]}_{pair[1]}"]
    print(f"[PID {os.getpid()}] Appending {pair} to log file at {filename_log}")
    with fs_local.open(filename_log, "w") as f:
        f.write('\n'.join(content) + "\n")


def get_existing_pairs(folder_results, filterid, component):
    fs = GCSFileSystem()
    filename = os.path.join(folder_results, f"logs/{filterid:02d}/stack_log_{filterid:02d}_{component}.txt")
    try:
        with fs.open(filename, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"No existing log file found at {filename}. Starting fresh.")
        content = ""
    if content:
        content = content.strip().split('\n')
        content = list(set(content))  # Remove duplicates
        content.sort()  # Sort for consistency
        print(f"{filterid} | {component} | {len(content)} Existing pairs in log: {content[0]} to {content[-1]}...")
    else:
        print("No existing pairs found in log.")

    content = [pair.split('_') for pair in content if pair.strip()]
    return content

def obspy_filter(data, sampling_rate, f_low, f_high):
    filtered = bandpass(data, freqmin=f_low, freqmax=f_high, df=sampling_rate, corners=4, zerophase=True)
    return filtered

def load_ccf_all_cloud(pairs, config, filterid_list, ref_start, ref_end, ccf_date_lst, sampling_rate=20, component_list=["ZZ"]):
    comp_idx = {"EE":0, "EN":1, "EZ":2, "NE":3, "NN":4, "NZ":5, "ZE":6, "ZN":7, "ZZ":8}
    comp_idx_list = [comp_idx[comp] for comp in component_list]

    pair_data = {tuple(pair): {"dates": [], "ccf": [], "ccf_ref": []} for pair in pairs}

    available_date_lst = []
    ccf_filt_allcomp_allfilter_allday = []
    ccf_ref_filt_allcomp_allfilter_allday = []

    ccf_len_anomaly_dic = {}
    for ccf_date in ccf_date_lst:
        year = ccf_date.year
        jday = ccf_date.dayofyear
        file = f'{config["folder_stacks_load"]}{year}/{year}.{jday:03d}.zarr/'

        date_int = int(f"{year}{jday:03d}")
        try:
            z_temp = zarr.open(f'gs://{file}', mode='r')
            st1_lst = z_temp['id1'][:]
            st2_lst = z_temp['id2'][:]

            pair_to_idx = {(".".join(st1.split(".")[:2]), ".".join(st2.split(".")[:2])): i for i, (st1, st2) in enumerate(zip(st1_lst, st2_lst))}
            xcorr = z_temp['xcorr']

        except FileNotFoundError:
            print(f"File not found: {file} at {year}.{jday}")
            continue
        except KeyError as e:
            print(f"Missing key {e} at {year}.{jday} in file {file}")
            continue

        log_head = f"STACKS | {config['project']} | {filterid_list} | {component_list} | {date_int}"

        for pair in pairs:
            try:
                pair_key = tuple(pair)
                idx = pair_to_idx.get(pair_key)
                if idx is None:
                    print(f"{log_head} | no pair {pair[0]}-{pair[1]} in [{file}].")
                    continue

                ccf = xcorr[idx, comp_idx_list, :]  # Assuming the first dimension is the pair index and the second is the component index

                # --- len check ---
                if ccf.shape[-1] != 12001:
                    print(f"{log_head} | Warning!!! CCF shape mismatch at {year}.{jday}, expected 12001 but got {ccf.shape}.")
                    ccf_len_anomaly_dic[date_int] = ccf.shape
                else:
                    ccf_filtered = np.asarray([obspy_filter(ccf, sampling_rate=sampling_rate, f_low=config["f_low_dic"][filterid], f_high=config["f_high_dic"][filterid]) for filterid in filterid_list])
                    pair_data[pair_key]["ccf"].append(ccf_filtered)
                    if (ccf_date >= ref_start) and (ccf_date <= ref_end):
                        pair_data[pair_key]["ccf_ref"].append(ccf_filtered)

                    pair_data[pair_key]["dates"].append(date_int)


            except KeyError as e:
                print(f"Missing key {e} at {year}.{jday} for pair: {pair}")
            except Exception as e:
                print(f"Error at {year}.{jday} for pair: {pair}: {type(e).__name__}: {e}")

    return pair_data

def sliding_stack_ccf(date_lst, ccf_all, half_window_days=2, min_traces=1, stack_method="mean"):
    """
    date_lst: list-like of datetime/date-like objects
    ccf_all:  2D array (N_days, N_lags)
    half_window_days: e.g. 2 for ±2 days
    min_traces: minimum number of CCFs required to form a stack
    stack_method: "mean" or "median"
    """
    dates = np.asarray([np.datetime64(datetime.strptime(str(date), "%Y%m%d").date()) for date in date_lst], dtype="datetime64[D]")
    ccf_all = np.asarray(ccf_all)
    n_days = len(dates)

    stacked = np.full_like(ccf_all, np.nan, dtype=np.result_type(ccf_all.dtype, np.float32))
    valid_mask = np.zeros(n_days, dtype=bool)
    delta = np.timedelta64(half_window_days, "D")

    for i, center in enumerate(dates):
        left = np.searchsorted(dates, center - delta, side="left")
        right = np.searchsorted(dates, center + delta, side="right")

        n_traces = right - left

        if n_traces < min_traces:
            continue

        window = ccf_all[left:right]

        if stack_method == "mean":
            stacked[i] = np.mean(window, axis=0)
        elif stack_method == "median":
            stacked[i] = np.median(window, axis=0)
        else:
            raise ValueError("stack_method must be 'mean' or 'median'")

        valid_mask[i] = True

    return stacked, valid_mask

def sliding_mean_stack_ccf(date_lst, ccf_all, half_window_days=2, min_traces=1):
    dates = np.asarray([np.datetime64(datetime.strptime(str(date), "%Y%j").date()) for date in date_lst], dtype="datetime64[D]")
    ccf_all = np.asarray(ccf_all)
    delta = np.timedelta64(half_window_days, "D")

    left = np.searchsorted(dates, dates - delta, side="left")
    right = np.searchsorted(dates, dates + delta, side="right")

    cumsum = np.vstack([np.zeros((1, ccf_all.shape[1]), dtype=np.float64),
                        np.cumsum(ccf_all, axis=0, dtype=np.float64)])

    counts = right - left

    stacked = np.full(ccf_all.shape, np.nan, dtype=np.float64)
    valid_mask = counts >= min_traces

    stacked[valid_mask] = (cumsum[right[valid_mask]] - cumsum[left[valid_mask]]) / counts[valid_mask, None]

    return stacked, valid_mask

def save_stacked_ccf(date, ccf_all, ccf_ref, store_path, local_mode=0):
    try:
        # --- Convert lists to NumPy arrays ---
        date_array = np.asarray(date)
        ccf_array = np.asarray(ccf_all)
        ccf_ref_array = np.asarray(ccf_ref)

        # -------------------------------------
        print(f"Save stacked result on {store_path}")
        if local_mode:
            fp = zarr.storage.LocalStore(
                store_path, read_only=False
            )
        else:
            fp = zarr.storage.FsspecStore.from_url(
                    store_path, read_only=False,
                    storage_options={"token": "google_default"}
                )

        zarr.create_array(fp, name="date", data=date_array, overwrite=True)
        zarr.create_array(fp, name="data", data=ccf_array, overwrite=True)
        zarr.create_array(fp, name="reference", data=ccf_ref_array, overwrite=True)

        return True
    except Exception as e:
        print(f"Error saving to {store_path}: {type(e).__name__}: {e}")
        return False

def save_stacked_ref_ccf(ccf_ref, ref_start, ref_end, store_path, local_mode=0):
    try:
        # --- Convert lists to NumPy arrays ---
        ccf_ref_time_array = np.asarray([ref_start, ref_end], dtype="str")
        ccf_ref_array = np.asarray(ccf_ref)

        # -------------------------------------
        print(f"Save stacked reference from {ref_start} to {ref_end} on {store_path}")
        if local_mode:
            fp = zarr.storage.LocalStore(
                store_path, read_only=False
            )
        else:
            fp = zarr.storage.FsspecStore.from_url(
                    store_path, read_only=False,
                    storage_options={"token": "google_default"}
                )
        zarr.create_array(fp, name="reference", data=ccf_ref_array, overwrite=True)
        zarr.create_array(fp, name="reference_time", data=ccf_ref_time_array, overwrite=True)
        return True
    except Exception as e:
        print(f"Error saving reference to {store_path}: {type(e).__name__}: {e}")
        return False

# ---------- define worker for one pair ----------
def process_pair_batch(pairs, config, filterid_list, ccf_date_lst, component_list=["ZZ"]):
    pair_id_dic = {""} # optional: define a mapping from station pairs to unique IDs
    ref_start = datetime.strptime(config["ref_start"], "%Y-%m-%d")
    ref_end = datetime.strptime(config["ref_end"], "%Y-%m-%d")
    sampling_rate = config["sampling_rate"]
    half_window_days = config["half_window_days"]
    folder_stacks = config["folder_stacks"]
    local_mode = config.get("local_mode", 0)

    pair_condition = {tuple(pair): {"status": "pending", "completed": [], "no_valid": []} for pair in pairs}

    pair_data = load_ccf_all_cloud(pairs, config, filterid_list, ref_start, ref_end, ccf_date_lst, sampling_rate, component_list)
    for pair in pairs:
        st1, st2 = pair
        result = pair_data.get(tuple(pair))
        available_date_lst = result["dates"]

        if not available_date_lst:
            print(f"[{st1}-{st2}] No available dates in specified range.")
            pair_condition[tuple(pair)]["status"] = "no_data"
            continue

        ccf_filt_lst_all = np.asarray(result["ccf"])
        ccf_ref_filt_lst = np.asarray(result["ccf_ref"])

        if len(ccf_ref_filt_lst) == 0:
            print(f"[{st1}-{st2}] Warning: no reference data in specified range")
            ccf_ref_all = np.zeros_like(ccf_filt_lst_all[0])  # Use zeros as fallback
        else:
            ccf_ref_all = np.nanmean(ccf_ref_filt_lst, axis=0)

        completed = []
        no_valid = []

        for i, filterid in enumerate(filterid_list):
            for j, component in enumerate(component_list):
                ccf_ref = ccf_ref_all[i, j]
                ccf_daily = ccf_filt_lst_all[:, i, j, :]

                stacked_ccf, valid_mask = sliding_mean_stack_ccf(available_date_lst,
                                                                 ccf_daily,
                                                                 half_window_days=half_window_days,
                                                                 min_traces=1)

                dates_valid = np.asarray(available_date_lst)[valid_mask]
                ccf_valid = stacked_ccf[valid_mask]

                if len(dates_valid) == 0:
                    no_valid.append((filterid, component))
                    continue

                if local_mode:
                    base = f"../../results/{folder_stacks}"
                else:
                    base = f"gs://{folder_stacks}"

                store_path = os.path.join(base, f'STACKS/{filterid:02d}/pm{half_window_days:03d}_DAYS/{component}/{st1}_{st2}.zarr/')
                store_ref_path = os.path.join(base, f'STACKS/{filterid:02d}/REF/{component}/{st1}_{st2}.zarr/')

                ok1 = save_stacked_ccf(dates_valid, ccf_valid, ccf_ref, store_path, local_mode=local_mode)
                ok2 = save_stacked_ref_ccf(ccf_ref, config["ref_start"], config["ref_end"], store_ref_path, local_mode=local_mode)
                if ok1 and ok2:
                    completed.append((filterid, component))

        pair_condition[tuple(pair)]["status"] = "done"
        pair_condition[tuple(pair)]["completed"] = completed
        pair_condition[tuple(pair)]["no_valid"] = no_valid

    return pair_condition

# ---------- parallel execution ----------
if __name__ == "__main__":
    args = parse_args()
    local_mode = args.local_mode
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    file_mark = args.file_mark

    print(f"Node rank: {node_rank}, Total nodes: {num_nodes}, File mark: {file_mark}, local_mode: {local_mode}")

    # --- import config based on project argument ---
    cfg = importlib.import_module(f"configs.{args.project}")

    station_pair_list = cfg.station_pair_list
    if station_pair_list.split('.')[-1] != 'txt':
        station_pair_list += file_mark + '.txt'

    folder_stacks_load = cfg.folder_stacks_load
    filterid_list = cfg.filterid_list
    component_list = cfg.component_list

    print(f"Load ccf from {folder_stacks_load} for filters {filterid_list} and components {component_list}")

    ccf_date_lst = pd.date_range(
    start=cfg.ccf_start,
    end=cfg.ccf_end,
    freq="D"
    ).to_list()

    config = {"project": cfg.project,
              "ref_start": cfg.ref_start,
              "ref_end": cfg.ref_end,
              "sampling_rate": cfg.sampling_rate,
              "half_window_days": cfg.half_window,
              "f_low_dic": cfg.f_low_dic,
              "f_high_dic": cfg.f_high_dic,
              "folder_stacks_load": folder_stacks_load,
              "folder_stacks": cfg.folder_results,
              "local_mode": local_mode  # Set to True for local testing
              }

    pairs_lst = get_pairs_from_file(station_pair_list)

    existing_pairs = []
    for filterid in filterid_list:
        for component in component_list:
            existing_pairs += get_existing_pairs(cfg.folder_results, filterid, component)

    pairs_lst_remain = []
    for i, pair in enumerate(pairs_lst):
        existing_n = existing_pairs.count(pair)
        if existing_n == len(filterid_list) * len(component_list):
            print(f"{i} [{pair[0]}-{pair[1]}] already processed for all filters and components, skipping.")
        else:
            print(f"{i} [{pair[0]}-{pair[1]}] has processed for {existing_n}/{len(filterid_list) * len(component_list)} cases.")
            pairs_lst_remain.append(pair)

    pairs_lst = pairs_lst_remain
    pairs_lst_to_process = pairs_lst[node_rank::num_nodes]
    print(f"{len(pairs_lst_to_process)} pairs in the list after node distribution: {pairs_lst_to_process}")
    print(f"{component_list = }")

    print(f"Pairs to process / Total: {len(pairs_lst_to_process)}/{len(pairs_lst)}")

    num_workers = os.cpu_count()  # tune this based on CPU / I/O / GCS rate limits

    PAIR_BATCH_SIZE = 1
    MAX_WORKERS = 2
    pair_batches = [pairs_lst_to_process[i:i + PAIR_BATCH_SIZE] for i in range(0, len(pairs_lst_to_process), PAIR_BATCH_SIZE)]
    total_tasks = len(pair_batches)
    max_workers = min(MAX_WORKERS, total_tasks)

    print(f"Processing {len(pairs_lst_to_process)} pairs in {len(pair_batches)} batches using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(process_pair_batch, batch, config, filterid_list, ccf_date_lst, component_list): batch for batch in pair_batches}

        for i, fut in enumerate(as_completed(futures), start=1):
            batch = futures[fut]
            try:
                 batch_result = fut.result()
            except Exception as e:
                print(f"Error processing batch {batch}: {type(e).__name__}: {e}")
                continue

            for pair_tuple, result in batch_result.items():
                print(f"[{pair_tuple[0]}-{pair_tuple[1]}] Status: {result['status']}, Completed: {len(result['completed'])}, No valid: {len(result['no_valid'])}")
                for filterid, component in result["completed"]:
                    log_progress(pair_tuple, config["folder_stacks"], filterid, component)

                for filterid, component in result["no_valid"]:
                    log_progress(pair_tuple, config["folder_stacks"], filterid, component, log_type="_no_valid")

                if result["status"] == "no_data":
                    for filterid in filterid_list:
                        for component in component_list:
                            log_progress(pair_tuple, config["folder_stacks"], filterid, component, log_type="_no_data")
