import os
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from igb_communication import igb_reader
import random
import json
from scipy.stats import qmc

# Global constants (using new values)


CHUNK_SIZE = 20_000

RANDOM_DELAY_ACTIVE = False
LOAD_DELAY = False
DELAY_MIN = 0.0
DELAY_MAX = 15.0

LOAD_FROM_SETTINGS = False

L0 = 50000.0
T0 = 215
V0 = 600

ECG0 = {
    "Lead I": 0.01562,
    "Lead II": 0.02538,
    "Lead III": 0.02247,
    "Lead aVR": 0.01931,
    "Lead aVL": 0.0185,
    "Lead aVF": 0.01904,
    "Lead V1": 0.2418,
    "Lead V2": 0.09964,
    "Lead V3": 0.3158,
    "Lead V4": 0.2313,
    "Lead V5": 0.1694,
    "Lead V6": 0.1251,
}
ECG_ORDER = [
    "Lead I", "Lead II", "Lead III", "Lead aVR", "Lead aVL", "Lead aVF",
    "Lead V1", "Lead V2", "Lead V3", "Lead V4", "Lead V5", "Lead V6"
]

################################################################################
# 1) Generate a single ECG for the entire simulation (downsample every 10 rows)
################################################################################


def read_real_ecg(ecg_csv_file, ecg_mask=None):
    df = pd.read_csv(ecg_csv_file, header=0)
    if ecg_mask is not None:
        df = df.iloc[ecg_mask].reset_index(drop=True)
    else:
        df = df / np.array([ECG0[lead] for lead in ECG_ORDER])
    return df.to_numpy(dtype=np.float32).T


################################################################################
# 2) Load node coordinates from the .pts file (common across simulations)
################################################################################


def load_points(points_file):
    with open(points_file, "r") as f:
        lines = f.readlines()
    pts = []
    for line in lines[1:]:
        x_str, y_str, z_str = line.split()
        pts.append((float(x_str), float(y_str), float(z_str)))
    return pts


################################################################################
# 3) Process activation data from act.igb for one simulation.
#    Builds a DataFrame with columns: x, y, z, T, sim_id.
################################################################################


def process_activation(sim_id, act_igb_file, points_file, delay=0.0):
    points = load_points(points_file)
    n_points = len(points)
    time_array, act_data, hdr = igb_reader(act_igb_file)
    T_raw = act_data[-1, :]  # Activation times for each node
    max_T = max(T_raw)
    if hdr["n_nodes"] != n_points:
        raise ValueError(
            "Mismatch in number of nodes between act.igb header and points file."
        )

    records = []
    for j in range(n_points):
        if random.randint(0, 100) < 90:
            continue
        x_raw, y_raw, z_raw = points[j]
        T_val = T_raw[j] + delay  # add the delay here
        T_val = T_val if T_val > -0.5 else max_T
        x_dim = x_raw / L0
        y_dim = y_raw / L0
        z_dim = z_raw / L0
        T_dim = T_val / T0
        record = {
            "x": np.float32(x_dim),
            "y": np.float32(y_dim),
            "z": np.float32(z_dim),
            "T": np.float32(round(T_dim, 4)),
            "sim_id": np.int32(sim_id),
        }
        records.append(record)
    return pd.DataFrame(records)


################################################################################
# 4) Process ECG data for one simulation.
#    Returns a DataFrame with columns: sim_id, ecg.
################################################################################


def get_settings(sim_id):
    with open("sim_raw/simulation_settings.json", "r") as f:
        settings = json.load(f)
        for setting in settings:
            print(setting)
            if int(setting["sim_id"]) == sim_id:
                return setting
        raise ValueError(f"Settings for sim_id {sim_id} not found.")


# Modify the ECG processing function to add delay:


def process_ecg(sim_id, ecg_csv_file, delay=0.0, ecg_mask=None, df_healthy=None):
    ecg_2d = read_real_ecg(ecg_csv_file, ecg_mask=ecg_mask)
    delay = delay if RANDOM_DELAY_ACTIVE else 0
    # ecg_2d = ecg_2d + delay  # add the delay to ECG times
    if RANDOM_DELAY_ACTIVE and LOAD_DELAY:
        ecg_2d = np.array([[delay / T0]], dtype=np.float32)
    if RANDOM_DELAY_ACTIVE and not LOAD_DELAY:
        round_delay = int(round(delay))
        # shift each lead by round_delay samples: pad with zeros at front and trim to original length
        n_leads, n_samples = ecg_2d.shape
        if round_delay > 0:
            pad = np.zeros((n_leads, round_delay), dtype=ecg_2d.dtype)
            ecg_2d = np.concatenate((pad, ecg_2d), axis=1)[:, :n_samples]

    if LOAD_FROM_SETTINGS:
        settings = get_settings(sim_id)
        parameters = [
            settings["fibrosis_center_x"],
            settings["fibrosis_center_y"],
            settings["fibrosis_center_z"],
            settings["fibrosis_size"],
        ]
        ecg_2d = np.array([parameters], dtype=np.float32)

    if df_healthy is not None:
        # If df_healthy is provided, use it to normalize the ECG
        ecg_2d = ecg_2d - df_healthy

    # Round all numbers in ecg_2d to the nearest 12 decimals
    ecg_2d = np.round(ecg_2d, decimals=12)

    return pd.DataFrame([{"sim_id": np.int32(sim_id), "ecg": ecg_2d.tolist()}])


################################################################################
# 5) Process all simulations and write two aggregated Parquet files:
#    1) activation_all.parquet and 2) ecg_all.parquet.
################################################################################


def process_all_simulations(sim_root, out_dir):
    sim_folders = [
        d for d in os.listdir(sim_root) if os.path.isdir(os.path.join(sim_root, d))
    ]
    activation_dfs = []
    ecg_dfs = []
    common_points_file = os.path.join(sim_root, "heart_1600.pts")
    # df_healthy = pd.read_csv("healthy_ecg.csv", header=0)
    # df_healthy = df_healthy / ECG0
    # df_healthy = df_healthy.to_numpy(dtype=np.float32).T
    df_healthy = None

    if RANDOM_DELAY_ACTIVE:
        new_folders = []
        for sim_id in sim_folders:
            for i in range(4):
                new_folders.append(sim_id)

        sim_folders = new_folders

    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=300)
    scaled_sample = qmc.scale(sample, l_bounds=[0], u_bounds=[78123])
    ecg_mask = sorted(int(x) for x in scaled_sample.flatten())
    ecg_mask = None
    i = 0
    for sim_id in sim_folders:

        sim_folder = os.path.join(sim_root, sim_id)
        print(f"Processing simulation {sim_id} in folder {sim_folder}")

        act_igb_file = os.path.join(sim_folder, "act.igb")
        ecg_csv_file = os.path.join(sim_folder, "ECG.csv")
        # cv_csv_file = os.path.join(sim_folder, "point_cv_stats.csv")

        delay = 0
        if RANDOM_DELAY_ACTIVE:
            delay = random.uniform(DELAY_MIN, DELAY_MAX)
        print(f"Using delay {delay:.4f} for simulation {sim_id}")

        data_sim_id = (
            int(sim_id) * 10 + int(i) % 10 if RANDOM_DELAY_ACTIVE else int(sim_id)
        )
        df_act = process_activation(
            data_sim_id, act_igb_file, common_points_file, delay=delay
        )
        activation_dfs.append(df_act)

        df_ecg = process_ecg(
            data_sim_id,
            ecg_csv_file,
            delay=delay,
            ecg_mask=ecg_mask,
            df_healthy=df_healthy,
        )
        ecg_dfs.append(df_ecg)
        i += 1

    df_all_act = pd.concat(activation_dfs, ignore_index=True)
    df_all_ecg = pd.concat(ecg_dfs, ignore_index=True)

    activation_parquet = os.path.join(out_dir, "activation_all.parquet")
    ecg_parquet = os.path.join(out_dir, "ecg_all.parquet")

    table_act = pa.Table.from_pandas(df_all_act)
    pq.write_table(table_act, activation_parquet)
    print(f"Activation data written to {activation_parquet}")

    table_ecg = pa.Table.from_pandas(df_all_ecg)
    pq.write_table(table_ecg, ecg_parquet)
    print(f"ECG data written to {ecg_parquet}")


################################################################################
# 6) Main entry point.
################################################################################


def main():
    sim_root = "sim_raw"  # Root folder containing simulation subfolders
    out_dir = "sim_parsed"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    process_all_simulations(sim_root, out_dir)


if __name__ == "__main__":
    main()
