from gcsfs.core import GCSFileSystem
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from obspy.signal.regression import linear_regression
from concurrent.futures import ProcessPoolExecutor, as_completed
import zarr

from args import parse_args
import importlib

args = parse_args()

def load_station_coords(station_info_file):
    """Load {(network, station): (longitude, latitude)} from a station_info.txt file
    (tab-separated columns: network, station, longitude, latitude)."""
    coords = {}
    with open(station_info_file, "r") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            network, station, lon, lat = parts
            coords[(network, station)] = (float(lon), float(lat))
    return coords


def haversine_distance_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in kilometers between two (lon, lat) points."""
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def parse_pair_stations(pair_name):
    """Split a 'NET1.STA1.LOC1.CHA1_NET2.STA2.LOC2.CHA2' pair name into its two
    (network, station) tuples."""
    part1, part2 = pair_name.split("_", 1)
    net1, sta1 = part1.split(".")[0], part1.split(".")[1]
    net2, sta2 = part2.split(".")[0], part2.split(".")[1]
    return (net1, sta1), (net2, sta2)


def get_distance_based_dtt_minlag(pair_name, station_coords, f_low_dic, filterid, dtt_v, default):
    """dtt_minlag = 3 * period + distance / dtt_v, where period comes from the lowest
    frequency of the current filter (f_low_dic[filterid]) and distance is the
    great-circle distance between the pair's two stations loaded from station_info.txt."""
    try:
        (net1, sta1), (net2, sta2) = parse_pair_stations(pair_name)
        lon1, lat1 = station_coords[(net1, sta1)]
        lon2, lat2 = station_coords[(net2, sta2)]
    except (ValueError, KeyError) as e:
        print(f"Could not resolve station coordinates for pair {pair_name} ({e}); using default dtt_minlag {default}")
        return default

    distance_km = haversine_distance_km(lon1, lat1, lon2, lat2)
    period = 1.0 / f_low_dic[filterid]
    return 3 * period + distance_km / dtt_v, distance_km


def get_file_list(config, component_mwcs):
    local_mode = config.get("local_mode", False)
    folder_mwcs = config["folder_project"]
    filterid_dtt = config["filterid"]
    half_window = config["half_window"]
    time_type = config["time_type"]
    mirror_ccf = config["mirror_ccf"]
    dtt_minlag_mode = config.get("dtt_minlag_mode", "file")

    mwcs_type = f"MWCS_{mirror_ccf}".replace("both", "unfold")

    path = f'../../results/{folder_mwcs}{mwcs_type}/{config["filterid"]:02d}/pm{config["half_window"]:03d}_{time_type}/{component}/'
    if not local_mode:
        path = path.replace("../../results/cctorch", "cctorch")
        fs = GCSFileSystem()
        fs.invalidate_cache("gs://")
        pair_dirs = fs.ls(path)

        if dtt_minlag_mode == "file":
            print(f"Loading dtt_minlag data from {file_lst_dtt}...")
            with fs.open(file_lst_dtt, "r") as f:
                dtt_minlag_data = {
                    line.strip().split()[0]: float(line.strip().split()[1])
                    for line in f
                }
        else:
            dtt_minlag_data = {}
    else:
        pair_dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        if dtt_minlag_mode == "file":
            print(f"Loading dtt_minlag data from {file_lst_dtt}...")
            with open(file_lst_dtt.replace('cctorch/ambient_noise_das/das_preprocess/dtt_info', 'configs/dtt_info'), "r") as f:
               dtt_minlag_data = {
                   line.strip().split()[0]: float(line.strip().split()[1])
                   for line in f.readlines()
                }
        else:
            dtt_minlag_data = {}

    print(f"Scanning MWCS results in {path} for DTT processing...")
    print(f"Total pair_dirs found: {len(pair_dirs)}")
    return pair_dirs, dtt_minlag_data

def get_targeting_pairs(station_pair_list):
    pairs = []
    with open(station_pair_list, "r") as f:
        pairs = [line.strip().replace(',', '_') for line in f if line.strip()]
    print(f"{len(pairs)} targeting pairs found.")
    return pairs

def get_existing_pairs(config, component):
    folder_results = config["folder_project"]
    filterid = config["filterid"]

    fs = GCSFileSystem()
    filename = os.path.join(folder_results, f"logs/{filterid:02d}/dtt_log_{filterid:02d}_{component}.txt")
    try:
        with fs.open(filename, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"No existing log file found at {filename}. Starting fresh.")
        content = ""
    if content:
        content = content.strip().split('\n')
        print(f"{len(content)} Existing pairs in log: {content}")
    else:
        print("No existing pairs found in log.")
    return content

def scan_avai_pair_date(pair_dir, local_mode=False):
    if local_mode:
        file_date_lst = [f for f in os.listdir(pair_dir) if f.endswith('.txt')]
    else:
        fs = GCSFileSystem()
        if pair_dir.startswith("gs://"):
            prefix = pair_dir
        else:
            prefix = f"gs://{pair_dir}"
        try:
            file_date_lst = fs.ls(prefix)
        except Exception as e:
            print(f"Failed to list {pair_dir}: {e}")
            return []

    date_lst = [date_str.split('/')[-1].split('.')[0] for date_str in file_date_lst if date_str.endswith('.txt')]

    print(f"{prefix}: Found {len(date_lst)} dates")
    return sorted(list(set(date_lst)))

def wavg_wstd(data, errors):
    d = data
    errors[errors == 0] = 1e-6

    w = 1. / errors
    wavg = (d * w).sum() / w.sum()
    N = len(np.nonzero(w)[0])
    wstd = np.sqrt(np.sum(w * (d - wavg) ** 2) / ((N - 1) * np.sum(w) / N))
    return wavg, wstd

def process_pair(pair_dir: str, config, comp, dtt_minlag_data):
    local_mode = config.get("local_mode", False)
    project = config["project"]
    file_dtt_out = config["folder_project"]
    dtt_minlag_default = config["dtt_minlag_default"]
    dtt_minlag_mode = config.get("dtt_minlag_mode", "file")
    dtt_v = config.get("dtt_v")
    f_low_dic = config.get("f_low_dic")
    station_coords = config.get("station_coords", {})
    mov_stacks = config["mov_stacks"]
    dtt_width = config["dtt_width"]
    dtt_sides = config["dtt_sides"]
    dtt_mincoh = config["dtt_mincoh"]
    dtt_maxerr = config["dtt_maxerr"]
    dtt_maxdt = config["dtt_maxdt"]
    filterid = config["filterid"]
    time_type = config["time_type"]
    half_window = config["half_window"]
    mirror_ccf = config["mirror_ccf"]

    mwcs_type = f"MWCS_{mirror_ccf}".replace("both", "unfold")

    pair_name = pair_dir.split('/')[-1].replace('.zarr', '')
    # print(f"=== Start pair_dir: {pair_dir} ===")

    # date_available = scan_avai_pair_date(pair_dir, local_mode=local_mode)
    # if not date_available:
    #     print(f"No available dates for pair_dir: {pair_dir}, skipping.")
    #     return

    if not os.path.exists(file_dtt_out):
        os.makedirs(file_dtt_out, exist_ok=True)

    fs_local = GCSFileSystem()
    if not fs_local.exists(file_dtt_out):
            fs_local.makedirs(file_dtt_out, exist_ok=True)



    if dtt_minlag_mode == "assigned":
        dtt_minlag = dtt_minlag_default
    elif dtt_minlag_mode == "distance":
        dtt_minlag, distance_km = get_distance_based_dtt_minlag(
            pair_name, station_coords, f_low_dic, filterid, dtt_v, default=dtt_minlag_default
        )
    else:  # "file"
        dtt_minlag = dtt_minlag_data.get(pair_name, dtt_minlag_default)
        if pair_name not in dtt_minlag_data:
            print(f"Using default dtt_minlag {dtt_minlag_default} for pair {pair_name}")

    comp_lst_dtt = [comp]

    for mov_stack in mov_stacks:
        for components in comp_lst_dtt:
            z_file = zarr.open(pair_dir, mode='r')
            date_available = z_file['date'][:]
            data = z_file['data'][:]

            output = os.path.join(
                file_dtt_out,
                'DTT',
                "%02i" % filterid,
                # f"%03i_{time_type}" % mov_stack,
                f"pm{half_window:03d}_{time_type}",
                components,
                pair_name)

            if local_mode:
                output = output.replace('cctorch', '../../results/cctorch')

            dvv_Dates = []
            dvv_M_lst = []
            dvv_M0_lst = []
            for i, current in enumerate(date_available):
                df = pd.DataFrame(
                                data[i],
                                columns=['t', 'dt', 'err', 'coh']
                            ).set_index('t')


                first = True

                # --- load info ---
                msg = f"DTT | {project} | {filterid:02d} | {components} | {pair_name} | {current} | dtt_minlag={dtt_minlag}"
                if dtt_minlag_mode == "distance":
                    msg += f' | {distance_km}km | 3T={f_low_dic[filterid]}s'
                print(msg)

                # --- set lag-time window for dtt ---
                tArray = df.index.values
                lmlag = -dtt_minlag
                rmlag = dtt_minlag
                lMlag = lmlag - dtt_width
                rMlag = rmlag + dtt_width

                if dtt_sides == "both":
                    tindex = np.where(((tArray >= lMlag) & (tArray <= lmlag)) | ((tArray >= rmlag) & (tArray <= rMlag)))[0]
                elif dtt_sides == "left":
                    tindex = np.where((tArray >= lMlag) & (tArray <= lmlag))[0]
                else:
                    tindex = np.where((tArray >= rmlag) & (tArray <= rMlag))[0]

                # --- initialize array ---
                tmp = np.setdiff1d(np.arange(len(tArray)),tindex)
                df.loc[df.index[tmp], 'err'] = 1.0
                df.loc[df.index[tmp], 'coh'] = 0.0
                if first:
                    tArray = df.index.values
                    dtArray = df['dt']
                    errArray = df['err']
                    cohArray = df['coh']
                    pairArray = [pair_name, ]
                    first = False
                else:
                    dtArray = np.vstack((dtArray, df['dt']))
                    errArray = np.vstack((errArray, df['err']))
                    cohArray = np.vstack((cohArray, df['coh']))
                    pairArray.append(pair_name)
                del df
                if not first:
                    #~ tindex = np.tindwhere(((tArray >= lMlag) & (tArray <= lmlag)) | (
                        #~ (tArray >= rmlag) & (tArray <= rMlag)))[0]

                    Dates = []
                    Pairs = []
                    M = []
                    EM = []
                    A = []
                    EA = []
                    M0 = []
                    EM0 = []
                    if len(pairArray) != 1:
                        # first stack all pairs to a ALL mean pair, using
                        # indexes of selected values:
                        new_dtArray = np.zeros(len(tArray))
                        new_errArray = np.zeros(len(tArray)) + 9999
                        new_cohArray = np.zeros(len(tArray))
                        for i in range(len(tArray)):
                            #~ if i in tindex:
                            if 1:
                                cohindex = np.where(
                                    cohArray[:, i] >= dtt_mincoh)[0]
                                errindex = np.where(
                                    errArray[:, i] <= dtt_maxerr)[0]
                                dtindex = np.where(
                                    np.abs(dtArray[:, i]) <= dtt_maxdt)[0]
                                index = np.intersect1d(cohindex, errindex)
                                index = np.intersect1d(index, dtindex)

                                wavg, wstd = wavg_wstd(
                                    dtArray[:, i][index],
                                    errArray[:, i][index])
                                new_dtArray[i] = wavg
                                new_errArray[i] = wstd
                                new_cohArray[i] = 1.0

                        dtArray = np.vstack((dtArray, new_dtArray))
                        errArray = np.vstack((errArray, new_errArray))
                        cohArray = np.vstack((cohArray, new_cohArray))
                        pairArray.append("ALL")
                        del new_cohArray, new_dtArray, new_errArray,\
                            cohindex, errindex, dtindex, wavg, wstd

                        # then stack selected pais to GROUPS:
                        groups = {}
                        npairs = len(pairArray)-1
                        for group in groups.keys():
                            new_dtArray = np.zeros(len(tArray))
                            new_errArray = np.zeros(len(tArray)) + 9999
                            new_cohArray = np.zeros(len(tArray))
                            pairindex = []
                            for j, pair in enumerate(pairArray[:npairs]):
                                net1, sta1, net2, sta2 = pair.split('_')
                                if sta1 in groups[group] and \
                                                sta2 in groups[group]:
                                    pairindex.append(j)
                            pairindex = np.array(pairindex)

                            for i in range(len(tArray)):
                                #~ if i in tindex:
                                if 1:
                                    cohindex = np.where(
                                        cohArray[:, i] >= dtt_mincoh)[0]
                                    errindex = np.where(
                                        errArray[:, i] <= dtt_maxerr)[0]
                                    dtindex = np.where(
                                        np.abs(dtArray[:, i]) <= dtt_maxdt)[0]


                                    index = np.intersect1d(cohindex,
                                                            errindex)
                                    index = np.intersect1d(index, dtindex)
                                    index = np.intersect1d(index, pairindex)


                                    wavg, wstd = wavg_wstd(
                                        dtArray[:, i][index],
                                        errArray[:, i][index])
                                    new_dtArray[i] = wavg
                                    new_errArray[i] = wstd
                                    new_cohArray[i] = 1.0

                            dtArray = np.vstack((dtArray, new_dtArray))
                            errArray = np.vstack((errArray, new_errArray))
                            cohArray = np.vstack((cohArray, new_cohArray))
                            pairArray.append(group)
                            del new_cohArray, new_dtArray, new_errArray,\
                                cohindex, errindex, dtindex, wavg, wstd
                            # END OF GROUP HANDLING


                    # then process all pairs + the ALL
                    if len(dtArray.shape) == 1:  # if there is only one pair:
                        dtArray = dtArray.values.reshape((1, dtArray.shape[0]))
                        cohArray = cohArray.values.reshape((1, cohArray.shape[0]))
                        errArray = errArray.values.reshape((1, errArray.shape[0]))

                    used = np.zeros(dtArray.shape)

                    for i, pair in enumerate(pairArray):
                        cohindex = np.where(cohArray[i] >= dtt_mincoh)[0]
                        errindex = np.where(errArray[i] <= dtt_maxerr)[0]
                        dtindex = np.where(np.abs(dtArray[i]) <= dtt_maxdt)[0]

                        #~ index = np.intersect1d(tindex, cohindex)
                        index = np.intersect1d(cohindex, errindex)
                        index = np.intersect1d(index, dtindex)

                        used[i][index] = 1.0

                        w = 1.0 / errArray[i][index]
                        w[~np.isfinite(w)] = 1.0
                        VecXfilt = tArray[index]
                        VecYfilt = dtArray[i][index]
                        if len(VecYfilt) >= 2:
                            m, a, em, ea = linear_regression(
                                VecXfilt, VecYfilt, w,
                                intercept_origin=False)
                            m0, em0 = linear_regression(
                                VecXfilt, VecYfilt, w,
                                intercept_origin=True)
                            M.append(m)
                            EM.append(em)
                            A.append(a)
                            EA.append(ea)

                            M0.append(m0)
                            EM0.append(em0)

                            Dates.append(current)
                            Pairs.append(pair)

                            del m, a, em, ea, m0, em0

                        del VecXfilt, VecYfilt, w
                        del index, cohindex, errindex, dtindex

                    # logger.debug(
                    #     "%s: exporting: %i pairs" % (current,
                    #                                     len(pairArray)))
                    df_out = pd.DataFrame(
                        {'Pairs': Pairs, 'M': M, 'EM': EM, 'A': A, 'EA': EA,
                            'M0': M0, 'EM0': EM0},
                        index=Dates)
                    # Needs to be changed !

                    # print(f"Saving DTT results to {output}")

                    fn = os.path.join(output, '%s.txt' % current)
                    if local_mode:
                        if not os.path.exists(output):
                            os.makedirs(output, exist_ok=True)

                        if os.path.exists(fn):
                            with open(fn, 'rt') as f_in:
                                existing = pd.read_csv(f_in, index_col="Pairs", parse_dates=True)

                            for _, row in df_out.iterrows():
                                if row.Pairs in existing.index.values:
                                    existing.drop(row.Pairs, inplace=True)
                                    # logger.debug("Pair: %s is already in the output file, overwriting" % row.Pairs)
                            existing["Pairs"] = existing.index.values
                            existing.set_index("Date", inplace=True)
                            df_out = pd.concat([df_out, existing])

                        with open(fn, 'wt') as f_out:
                            df_out.to_csv(f_out, index_label='Date')

                    else:
                        if not fs_local.exists(output):
                            fs_local.makedirs(output, exist_ok=True)

                        if fs_local.exists(fn):
                            with fs_local.open(fn, 'rt') as f_in:
                                existing = pd.read_csv(f_in, index_col="Pairs", parse_dates=True)

                            for _, row in df_out.iterrows():
                                if row.Pairs in existing.index.values:
                                    existing.drop(row.Pairs, inplace=True)
                                    # logger.debug("Pair: %s is already in the output file, overwriting" % row.Pairs)
                            existing["Pairs"] = existing.index.values
                            existing.set_index("Date", inplace=True)
                            df_out = pd.concat([df_out, existing])

                        with fs_local.open(fn, 'wt') as gcs_file:
                            df_out.to_csv(gcs_file, index_label='Date')
                    # print(f"{pair_dir}: DTT results saved to {fn}")

                    if len(M) > 0 and len(M0) > 0:
                        dvv_Dates.append(Dates[0])
                        dvv_M_lst.append(-1*M[0])
                        dvv_M0_lst.append(-1*M0[0])

                    del df_out, M, EM, A, EA, M0, EM0, Pairs, Dates, used
                    del tArray, dtArray, errArray, cohArray, pairArray
            del output

            output = os.path.join(
            file_dtt_out,
            'DVV',
            "%02i" % filterid,
            #  f"%03i_{time_type}" % mov_stack,
            f"pm{half_window:03d}_{time_type}",
            components)

            if local_mode:
                output = output.replace('cctorch', '../../results/cctorch')

            df_out = pd.DataFrame(
                                {'dvv_M': dvv_M_lst, 'dvv_M0': dvv_M0_lst},
                                index=dvv_Dates)

            fn = os.path.join(output, '%s.txt' % pair_name)
            if local_mode:
                if not os.path.exists(output):
                    os.makedirs(output, exist_ok=True)

                with open(fn, 'wt') as f_out:
                    df_out.to_csv(f_out, index_label='Date')
            else:
                if not fs_local.exists(output):
                    print(f"Creating directory {output} for DVV results...")
                    fs_local.makedirs(output, exist_ok=True)

                with fs_local.open(fn, 'wt') as gcs_file:
                    df_out.to_csv(gcs_file, index_label='Date')

            del output, dvv_M_lst, dvv_M0_lst, dvv_Dates



            # --- write to log file ---
            output_log = os.path.join(
                file_dtt_out,
                mwcs_type,
                "%02i" % filterid,
                # f"%03i_{time_type}" % mov_stack,
                f"pm{half_window:03d}_{time_type}",
                comp,
                pair_name)

            filename_log = os.path.join(file_dtt_out, f"logs/{filterid:02d}/dtt_log_{filterid:02d}_{comp}.txt")
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

            content += [output_log]
            print(f"[PID {os.getpid()}] Appending {output_log} to log file at {filename_log}")
            with fs_local.open(filename_log, "w") as f:
                f.write('\n'.join(content) + "\n")


if __name__ == "__main__":
    node_rank = args.node_rank
    num_nodes = args.num_nodes
    filterid_dtt = args.filterid_dtt
    component = args.component
    local_mode = args.local_mode
    print(f"Starting DTT with node_rank={node_rank}, num_nodes={num_nodes}, filterid_dtt={filterid_dtt}, component={component}")

    cfg = importlib.import_module(f"configs.{args.project}")
    component_mwcs = component
    # file_list_path = cfg.mwcs_paths
    file_lst_dtt = cfg.file_dtt_minlag
    station_pair_list = cfg.station_pair_list_mwcs

    # dtt_minlag_mode: "assigned" (always use dtt_minlag_default), "file" (look up
    # per-pair values from file_dtt_minlag), or "distance" (derive it from the
    # inter-station distance in station_info.txt together with dtt_v/f_low_dic).
    # Existing configs only set the legacy boolean dynamic_dtt_minlag, which maps
    # to "assigned" (True) or "file" (False); set dtt_minlag_mode = "distance" in
    # a project config to opt into the new distance-based mode.
    dtt_minlag_mode = getattr(cfg, "dtt_minlag_mode", "assigned" if cfg.dynamic_dtt_minlag else "file")
    station_info_file = getattr(cfg, "station_info_file", "configs/dtt_info/station_info.txt")
    station_coords = load_station_coords(station_info_file) if dtt_minlag_mode == "distance" else {}

    config = {"project": cfg.project,
              "folder_project": cfg.folder_results,
              "dtt_minlag_default": cfg.dtt_minlag_default,
              "mov_stacks": cfg.mov_stacks,
              "dtt_width": cfg.dtt_width,
              "dtt_sides": cfg.dtt_sides,
              "dtt_mincoh": cfg.dtt_mincoh,
              "dtt_maxerr": cfg.dtt_maxerr,
              "dtt_maxdt": cfg.dtt_maxdt,
              "filterid": filterid_dtt,
              "half_window": cfg.half_window,
              "time_type": cfg.time_type,
              "mirror_ccf": cfg.mirror_ccf,
              "local_mode": local_mode,
              "dtt_minlag_mode": dtt_minlag_mode,
              "dtt_v": cfg.dtt_v,
              "f_low_dic": cfg.f_low_dic,
              "station_coords": station_coords}

    print(f"Configuration for DTT: {config}")


    pair_dirs, dtt_minlag_data = get_file_list(config, component_mwcs)
    pairs = get_targeting_pairs(station_pair_list)
    pair_dirs = [p for p in pair_dirs if any(pair in p for pair in pairs)]
    print(f"{len(pair_dirs)} pair_dirs in the list fitting with the targeting pairs.")

    existing_pairs = get_existing_pairs(config, component_mwcs)
    pair_dirs = [pair_dir for pair_dir in pair_dirs if pair_dir not in existing_pairs]
    print(f"{len(pair_dirs)} pair_dirs in the list after excluding existing pairs: {pair_dirs}")

    pair_dirs_to_process = pair_dirs[node_rank::num_nodes]
    print(f"{len(pair_dirs_to_process)} pair_dirs in the list after node distribution: {pair_dirs_to_process}")

    print(f"Process Pair / Existing Pairs / Total: {len(pair_dirs_to_process)} / {len(existing_pairs)} / {len(pairs)}")

    max_workers = min(len(pair_dirs_to_process), os.cpu_count() or 4)
    print(f"Using {max_workers} workers for {len(pair_dirs_to_process)} pair_dirs with component {component_mwcs} on node_rank {node_rank}/{num_nodes}")


    # --- process pairs with single thread ---
    for p in pair_dirs_to_process:
        process_pair(p, config, component_mwcs, dtt_minlag_data)

    # --- process pairs in parallel ---
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(process_pair, p, config, component_mwcs, dtt_minlag_data): p for p in pair_dirs_to_process}
    #     for fut in as_completed(futures):
    #         p = futures[fut]
    #         try:
    #             fut.result()
    #         except Exception as e:
    #             print(f"Error in pair_dir {p}: {e}")

    print(f"Node rank {node_rank} completed processing {len(pair_dirs_to_process)} pair_dirs.")
