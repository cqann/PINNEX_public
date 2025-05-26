import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from igb_communication import igb_reader

L0 = 20000.0
T0 = 60.0

CHUNK_SIZE = 20_000  # How many rows to accumulate before writing spatiotemporal data


################################################################################
# 1) Generate a single ECG for the entire simulation
################################################################################


def read_real_ecg(ecg_csv_file):
    df = pd.read_csv(ecg_csv_file, header=None)
    df = df.iloc[::10]
    return df.to_numpy(dtype=np.float32).T

################################################################################
# 2) Load node coordinates from the .pts file
################################################################################


def load_points(points_file):
    """
    Reads the .pts file, returning a list of (x, y, z).
    Assumes the first line is a header to skip.
    """
    with open(points_file, "r") as f:
        lines = f.readlines()

    pts = []
    for line in lines[1:]:  # skip the first header line
        x_str, y_str, z_str = line.split()
        x, y, z = float(x_str), float(y_str), float(z_str)
        pts.append((x, y, z))
    return pts

################################################################################
# 3) Stream parse voltage files, chunking spatiotemporal data to Parquet
################################################################################


def stream_parse_and_write_activation(
    act_igb_file,
    points_file,
    out_parquet_path,
    sim_id=1
):
    points = load_points(points_file)
    n_points = len(points)

    time_array, act_data, hdr = igb_reader(act_igb_file)
    T_raw = act_data[-1, :]  # shape: (N,) – activation times for each node

    # Sanity check: number of nodes must match
    if hdr["n_nodes"] != n_points:
        raise ValueError("Mismatch in number of nodes between act.igb header and points file.")

    arrow_schema = pa.schema([
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
        pa.field("z", pa.float32()),
        pa.field("T", pa.float32()),
        pa.field("sim_id", pa.int32()),
    ])

    writer = None
    chunk_records = []

    for j in range(n_points):
        x_raw, y_raw, z_raw = points[j]
        T_val = T_raw[j]

        x_dim = x_raw / L0
        y_dim = y_raw / L0
        z_dim = z_raw / L0
        T_dim = T_val / T0  # dimensionless activation time if desired

        record = {
            "x": np.float32(x_dim),
            "y": np.float32(y_dim),
            "z": np.float32(z_dim),
            "T": np.float32(round(T_dim, 4)),
            "sim_id": np.int32(sim_id),
        }
        chunk_records.append(record)

        # Write in chunks
        if len(chunk_records) >= CHUNK_SIZE:
            df_chunk = pd.DataFrame(chunk_records)
            table = pa.Table.from_pandas(df_chunk, schema=arrow_schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet_path, table.schema)
            writer.write_table(table)
            chunk_records = []

    # Write any leftover rows
    if chunk_records:
        df_chunk = pd.DataFrame(chunk_records)
        table = pa.Table.from_pandas(df_chunk, schema=arrow_schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()
    print(f"Activation data written to {out_parquet_path}")

################################################################################
# 4) Write the (sim_id, ecg) mapping to a separate parquet file
################################################################################


def write_ecg_parquet(sim_id, ecg_2d, out_parquet_path):
    """
    Saves a single row: { sim_id, ecg } to a separate parquet file,
    with ecg stored as a 2D nested list [list of list of float32].
    """
    # ecg_2d is shape (n_leads, seq_len)
    # Convert to nested Python list, each row is float32
    ecg_nested_list = ecg_2d.astype(np.float32).tolist()

    df_ecg = pd.DataFrame([
        {"sim_id": sim_id, "ecg": ecg_nested_list}
    ])

    # arrow schema for nested ecg: list(list(float32))
    arrow_schema_ecg = pa.schema([
        pa.field("sim_id", pa.int32()),
        pa.field("ecg", pa.list_(pa.list_(pa.float32()))),  # Nested list of float32
    ])

    table = pa.Table.from_pandas(df_ecg, schema=arrow_schema_ecg, preserve_index=False)
    pq.write_table(table, out_parquet_path)
    print(f"ECG data (sim_id={sim_id}) written to {out_parquet_path}")

################################################################################
# 5) Main
################################################################################


def main():
    # Input IGB file paths
    act_igb_file = "data/synthetic/raw/act_cube.igb"
    points_file = "data/synthetic/raw/block_cube.pts"

    # Output file paths
    spatiotemp_parquet = "data/synthetic/parsed/spatio_EIK_vol0.parquet"
    ecg_parquet = "data/synthetic/parsed/ecg_EIK_vol0.parquet"

    sim_id = 1

    stream_parse_and_write_activation(
        act_igb_file=act_igb_file,
        points_file=points_file,
        out_parquet_path=spatiotemp_parquet,
        sim_id=sim_id
    )

    ecg_csv_file = "data/synthetic/raw/healthy_ECG.csv"
    ecg_2d = read_real_ecg(ecg_csv_file)
    write_ecg_parquet(sim_id, ecg_2d, ecg_parquet)


if __name__ == "__main__":
    main()
