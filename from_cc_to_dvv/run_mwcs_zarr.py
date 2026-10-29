import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

from datetime import datetime, timedelta
import numpy as np
import zarr
from obspy import read
from gcsfs.core import GCSFileSystem
from matplotlib import pyplot as plt

from obspy.signal.invsim import cosine_taper
from obspy.signal.regression import linear_regression
from obspy.signal.filter import bandpass
import scipy
import scipy.fft as sf

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging

from args import parse_args
import importlib



args = parse_args()

def get_file_list(config, time_type, component):
    local_mode = config.get("local_mode", False)
    path = f'../../results/{config["folder_results"]}STACKS/{config["filterid_mwcs"]:02d}/pm{config["half_window"]:03d}_{time_type}/{component}/'
    if not local_mode:
        path = path.replace("../../results/", "")

        fs = GCSFileSystem()
        fs.invalidate_cache(config["path_gs"])
        file_lst = fs.ls(path)
        file_lst.sort()

    else:
        file_lst = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.zarr')]
        file_lst.sort()

    print(f"{len(file_lst)} files in the list in total.")
    return file_lst

def get_targeting_pairs(station_pair_list):
    pairs = []
    with open(station_pair_list, "r") as f:
        pairs = [line.strip().replace(',', '_') for line in f if line.strip()]
    print(f"{len(pairs)} targeting pairs found.")
    return pairs

def get_existing_pairs(folder_results, filterid_mwcs, component):
    fs = GCSFileSystem()
    filename = os.path.join(folder_results, f"logs/{filterid_mwcs:02d}/mwcs_log_{filterid_mwcs:02d}_{component}.txt")
    try:
        with fs.open(filename, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"No existing log file found at {filename}. Starting fresh.")
        content = ""
    if content:
        content = content.strip().split('\n')
        content = [file.replace('MWCS_unfold', 'STACKS') for file in content]
        content = [f"{file}.zarr" for file in content]
        print(f"{len(content)} existing pairs in log: {content}")
    else:
        print("No existing pairs found in log.")
    return content

def nextpow2(x):
    return np.ceil(np.log2(np.abs(x)))

def smooth(x, window='boxcar', half_win=3):
    """ some window smoothing """
    # TODO: docsting
    window_len = 2 * half_win + 1
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    if window == "boxcar":
        w = scipy.signal.windows.boxcar(window_len).astype('complex')
    else:
        w = scipy.signal.windows.hann(window_len).astype('complex')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[half_win:len(y) - half_win]

def getCoherence(dcs, ds1, ds2):
    # TODO: docsting
    n = len(dcs)
    coh = np.zeros(n).astype('complex')
    valids = np.argwhere(np.logical_and(np.abs(ds1) > 0, np.abs(ds2 > 0)))
    coh[valids] = dcs[valids] / (ds1[valids] * ds2[valids])
    coh[coh > (1.0 + 0j)] = 1.0 + 0j
    return coh

def mwcs(current, reference, freqmin, freqmax, df, tmin, window_length, step,
         smoothing_half_win=5):

    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    window_length_samples = int(window_length * df)
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.85)
    minind = 0
    maxind = window_length_samples
    while maxind <= len(current):
        cci = current[minind:(minind + window_length_samples)]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = reference[minind:(minind + window_length_samples)]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(step*df)
        maxind += int(step*df)

        fcur = sf.fft(cci, n=padd)[:padd // 2]
        fref = sf.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # Calculate the cross-spectrum
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(smooth(fcur2, window='hanning',
                                  half_win=smoothing_half_win))
            dref = np.sqrt(smooth(fref2, window='hanning',
                                  half_win=smoothing_half_win))
            X = smooth(X, window='hanning',
                       half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = sf.fftfreq(len(X) * 2, 1. / df)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= freqmin,
                                                 freq_vec <= freqmax))

        # Get Coherence and its mean value
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin
        # weights for the WLS must be the variance !
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())

        delta_t.append(m)

        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+window_length/2.+count*step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(current) + step*df:
        logging.warning("The last window was too small, but was computed")

    return np.array([time_axis, delta_t, delta_err, delta_mcoh]).T


def load_stacked_ref_ccf(file, time_type):
    print(f'Loading {file} ...')
    file_name = file.split('/')[-1].split('.zarr')[0]
    st1, st2 = file_name.split('_')

    z = zarr.open(f'gs://{file}')

    date_lst = z['date'][:]
    ccf_ref = z['reference'][:]
    ccf_all = z['data'][:]

    date_lst = [str(date) for date in date_lst]
    if time_type == "DAYS":
        date_lst = [f"{date[:4]}-{date[4:6]}-{date[6:]}" for date in date_lst]
    elif time_type == "HOURS":
        date_lst = [f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}" for date in date_lst]

    return st1, st2, date_lst, ccf_all, ccf_ref

def load_stacked_ref_ccf_v2(file, time_type):
    print(f'Loading {file} ...')
    file_name = file.split('/')[-1].split('.zarr')[0]
    st1, st2 = file_name.split('_')

    z = zarr.open(file)

    date_lst = z['date'][:]
    # ccf_ref = z['reference'][:]
    ccf_all = z['data'][:]

    # date_lst = [str(date) for date in date_lst]
    # if time_type == "DAYS":
    #     date_lst = [f"{date[:4]}-{date[4:6]}-{date[6:]}" for date in date_lst]
    # elif time_type == "HOURS":
    #     date_lst = [f"{date[:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}" for date in date_lst]

    return st1, st2, date_lst, ccf_all, None

def load_ref_ccf(file):
    print(f'Loading reference CCF from {file} ...')
    z = zarr.open(file)
    ccf_ref = z['reference'][:]
    return ccf_ref

def save_stacked_mwcs(date, mwcs_all, store_path, local_mode=0):
    try:
        # --- Convert lists to NumPy arrays ---
        date_array = np.asarray(date)
        mwcs_array = np.asarray(mwcs_all)

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
        zarr.create_array(fp, name="data", data=mwcs_array, overwrite=True)

        return True
    except Exception as e:
        print(f"Error saving to {store_path}: {type(e).__name__}: {e}")
        return False

def process_file(file, config, component):
    filterid_mwcs = config["filterid_mwcs"]
    sampling_rate = config["sampling_rate"]
    mwcs_low = config["mwcs_low"]
    mwcs_high = config["mwcs_high"]
    maxlag = config["maxlag"]
    mwcs_wlen = config["mwcs_wlen"]
    mwcs_step = config["mwcs_step"]
    folder_results = config["folder_results"]
    time_type = config["time_type"]
    half_window = config["half_window"]
    mirror_ccf = config["mirror_ccf"]
    local_mode = config.get("local_mode", 0)

    if not local_mode:
        file.replace('cctorch', 'gs://cctorch')

    fs_local = GCSFileSystem()
    st1, st2, date_lst, ccf_all_stacked, _ = load_stacked_ref_ccf_v2(file, time_type)
    ccf_ref = load_ref_ccf(file.replace(f"pm{half_window:03d}_{time_type}", "REF"))
    if (mirror_ccf == 'left') or (mirror_ccf == 'right'):
        ccf_ref = ccf_ref[::-1]
        mwcs_type = f"MWCS_{mirror_ccf}"
    else:
        mwcs_type = "MWCS_unfold"

    # === mwcs ===
    ref_name = f'{st1}_{st2}'
    print(f"[PID {os.getpid()}] Processing pair {ref_name} with "
          f"mwcs_low={mwcs_low}, mwcs_high={mwcs_high}, maxlag={maxlag}, "
          f"mwcs_wlen={mwcs_wlen}, mwcs_step={mwcs_step}, mirror_ccf={mirror_ccf}")

    outfolder = os.path.join(
            folder_results,
            mwcs_type,
            "%02i" % filterid_mwcs,
            # f"%03i_{time_type}" % mov_stack,
            f"pm{half_window:03d}_{time_type}",
            component,
            ref_name)

    if not fs_local.exists(outfolder):
        fs_local.makedirs(outfolder, exist_ok=True)

    def process_one_date(idx_date):
        i, date = idx_date
        # print(f"[process_one_date] {i} | date: {date}")
        # print(f"[process_one_date] {i} | mwcs_low: {mwcs_low}, mwcs_high: {mwcs_high}, sampling_rate: {sampling_rate}, maxlag: {maxlag}, mwcs_wlen: {mwcs_wlen}, mwcs_step: {mwcs_step}")
        # print(f"[process_one_date] {i} | ccf_ref: {ccf_ref[:10]}")
        # print(f"[process_one_date] {i} | ccf_all_stacked[i]: {ccf_all_stacked[i][:10]}")

        ccf_current = ccf_all_stacked[i]
        n = len(ccf_current)
        if mirror_ccf == 'left':
            ccf_current = ccf_current[::-1]
            first_half = ccf_current[:n//2].copy()
            ccf_current[n//2+n%2:] = first_half[::-1]
        elif mirror_ccf == 'right':
            ccf_current = ccf_current[::-1]
            second_half = ccf_current[n//2+n%2:].copy()[::-1]
            ccf_current[:n//2] = second_half

        output = mwcs(
            current=ccf_current,
            reference=ccf_ref,
            freqmin=mwcs_low,
            freqmax=mwcs_high,
            df=sampling_rate,
            tmin=-maxlag,
            window_length=mwcs_wlen,
            step=mwcs_step
            )

        # filename = os.path.join(outfolder, "%s.txt" % str(date))

        # with fs_local.open(filename, 'wt') as gcs_file:
        #     np.savetxt(gcs_file, output, fmt='%.16g')

        print(f"MWCS | {filterid_mwcs:02d} | {component} | {mirror_ccf} | {date} | {file}")
        return i, date, output

    inner_workers = min(1, len(date_lst))
    print(f"[PID {os.getpid()}] Parallelizing {len(date_lst)} dates with {inner_workers} threads")

    results = [None] * len(date_lst)

    with ThreadPoolExecutor(max_workers=inner_workers) as executor:
        futures = {
            executor.submit(process_one_date, (i, date)): (i, date)
            for i, date in enumerate(date_lst)
        }

        for fut in as_completed(futures):
            (i, date) = futures[fut]
            try:
                i_result, date_result, output = fut.result()
                results[i_result] = (date_result, output)

            except Exception as e:
                print(f"[PID {os.getpid()}] ❌ Error on {ref_name} date {date}: {e}")

    output_all = np.stack([output for date, output in results if output is not None], axis=0)

    if local_mode:
        base = f"../../results/{folder_results}"
    else:
        base = f"gs://{folder_results}"

    store_path = os.path.join(base, f'{mwcs_type}/{filterid_mwcs:02d}/pm{half_window:03d}_{time_type}/{component}/{st1}_{st2}.zarr/')
    save_stacked_mwcs(date_lst, output_all, store_path, local_mode=local_mode)

    # --- write to log file ---
    filename_log = os.path.join(folder_results, f"logs/{filterid_mwcs:02d}/mwcs_log_{filterid_mwcs:02d}_{component}.txt")
    try:
        with fs_local.open(filename_log, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"No existing log file found at {filename_log}. Starting fresh.")
        content = []

    if content:
        print(f"Existing log file content at {filename_log}: {content}")
        content = content.strip().split('\n')
        content = list(set(content))  # Remove duplicates
        content.sort()

    content += [outfolder.replace(mwcs_type, 'STACKS')]
    print(f"[PID {os.getpid()}] Appending {outfolder} to log file at {filename_log}")
    with fs_local.open(filename_log, "w") as f:
        f.write('\n'.join(content) + "\n")

    return ref_name

def main(node_rank, num_nodes, filterid_mwcs, component, local_mode):
    cfg = importlib.import_module(f"configs.{args.project}")
    path_gs = cfg.path_gs
    time_type = cfg.time_type
    station_pair_list = cfg.station_pair_list_mwcs
    folder_results = cfg.folder_results

    config = {"project": cfg.project,
              "path_gs": path_gs,
              "filterid_mwcs": filterid_mwcs,
              "half_window": cfg.half_window,
              "sampling_rate": cfg.sampling_rate,
              "mwcs_low": cfg.mwcs_low_dic[filterid_mwcs],
              "mwcs_high": cfg.mwcs_high_dic[filterid_mwcs],
              "maxlag": cfg.maxlag,
              "mwcs_wlen": cfg.mwcs_wlen_dic[filterid_mwcs],
              "mwcs_step": cfg.mwcs_step_dic[filterid_mwcs],
              "folder_results": folder_results,
              "time_type": time_type,
              "mirror_ccf": cfg.mirror_ccf,
              "local_mode": local_mode}

    print(f"Configuration for MWCS: {config}")


    file_lst = get_file_list(config, time_type, component)  # GCS call done
    pairs = get_targeting_pairs(station_pair_list)
    file_lst = [file for file in file_lst if any(pair in file for pair in pairs)]
    print(f"{len(file_lst)} files in the list fitting with the targeting pairs.")

    #existing_pairs = get_existing_pairs(folder_results, filterid_mwcs, component)
    #file_lst = [file for file in file_lst if file.replace('../../results/cctorch', 'cctorch') not in existing_pairs]
    print(f"{len(file_lst)} files in the list after excluding existing pairs: {file_lst}")

    files_to_process = file_lst[node_rank::num_nodes]
    print(f"{len(files_to_process)} files in the list after node distribution: {files_to_process}")

    #print(f"Process Files / Existing Pairs / Total: {len(files_to_process)} / {len(existing_pairs)} / {len(file_lst)}")

    # max_workers = os.cpu_count() or 1
    # print(f"Using {max_workers} processes (outer level)")

    pair_lst_mwcs = []

    # --- single-threaded execution for debugging ---
    for file in files_to_process:
        try:
            pair_name = process_file(file, config, component)
            pair_lst_mwcs.append(pair_name)
            print(f"✅ Finished pair {pair_name} from file {file}")

        except Exception as e:
            print(f"❌ Error processing file {file}: {e}")

    # --- multiprocessing with inner threading for date-level parallelism ---
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     future_to_file = {
    #         executor.submit(process_file, f, config, component): f for f in files_to_process
    #     }

    #     for future in as_completed(future_to_file):
    #         file = future_to_file[future]
    #         try:
    #             pair_name = future.result()
    #             pair_lst_mwcs.append(pair_name)
    #             print(f"✅ Finished pair {pair_name} from file {file}")
    #         except Exception as e:
    #             print(f"❌ Error processing file {file}: {e}")

    print("All done.")
    print(f"Total processed pairs: {len(pair_lst_mwcs)}")


if __name__ == "__main__":
    local_mode = args.local_mode
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    filterid_mwcs = args.filterid_mwcs
    component = args.component
    print(f"Starting MWCS with node_rank={node_rank}, num_nodes={num_nodes}, filterid_mwcs={filterid_mwcs}, component={component}, local_mode: {local_mode}")

    main(node_rank, num_nodes, filterid_mwcs, component, local_mode)
